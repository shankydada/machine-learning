import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import gdown

# Define custom transformer for outlier handling (with updates)
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)  # Ensure X is a DataFrame
        for col in X.select_dtypes(include=np.number).columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.factor * iqr
            upper_bound = q3 + self.factor * iqr
            X[col] = X[col].clip(lower_bound, upper_bound)
        return X.to_numpy()  # Ensure compatibility with scikit-learn

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else []

# Download dataset from Google Drive
file_id = '1cpk8TH3oLtaYUjKqfGC_QIu1NeF9Fuij'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'data.csv'
gdown.download(url, output, quiet=False)

# Load the dataset
df = pd.read_csv(output)

# Data preprocessing
df['Deaths'] = df['Deaths'].fillna(df['Deaths'].median())
df['Long_'] = df['Long_'].fillna(df['Long_'].median())
df['Lat'] = df['Lat'].fillna(df['Lat'].median())
df['Case_Fatality_Ratio'] = df['Case_Fatality_Ratio'].fillna(df['Case_Fatality_Ratio'].median())  # Fixed typo

# Features and target variables
X = df.drop(columns=['Case_Fatality_Ratio'])
y = df['Case_Fatality_Ratio']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model pipeline
pipeline = Pipeline([
    ('outlier_handler', OutlierHandler(factor=1.5)),  # Handle outliers
    ('scaler', StandardScaler()),                    # Scale the data
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),  # Add interaction terms
    ('feature_selection', SelectKBest(score_func=f_regression, k='all')),  # Select important features
    ('model', XGBRegressor(objective='reg:squarederror', random_state=42))  # Use XGBoost
])

# Hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Advanced Model Evaluation:\nRMSE: {rmse:.2f}\nR2 Score: {r2:.2f}\n")
print(f"Best Parameters: {grid_search.best_params_}")

# Save the best model
joblib.dump(best_model, 'Advanced_Case_Fatality_Ratio.pkl')
print("Advanced model saved successfully")

# Example prediction for new data
new_data = pd.DataFrame({
    'Lat': [7.5, 8.2],
    'Long_': [-13.2, -10.8],
    'Deaths': [150, 200],
})

y_pred = best_model.predict(new_data)
print('Case_Fatality_Ratio Predictions:', y_pred)
