from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load and preprocess data (replace with your actual data loading and preprocessing)
data = pd.DataFrame({
    'latitude': np.random.uniform(-90, 90, 100),
    'longitude': np.random.uniform(-180, 180, 100),
    'population': np.random.randint(1000, 1000000, 100),
    'deaths': np.random.randint(0, 1000, 100),
    'confirmed_cases': np.random.randint(0, 10000, 100),
    'case_fatality_ratio': np.random.uniform(0, 0.1, 100)
})

X = data[['latitude', 'longitude', 'population']]
y_deaths = data['deaths']
y_cases = data['confirmed_cases']
y_cfr = data['case_fatality_ratio']

imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

rf_deaths = RandomForestRegressor(n_estimators=100, random_state=42)
rf_cases = RandomForestRegressor(n_estimators=100, random_state=42)
rf_cfr = RandomForestRegressor(n_estimators=100, random_state=42)

rf_deaths.fit(X_scaled, y_deaths)
rf_cases.fit(X_scaled, y_cases)
rf_cfr.fit(X_scaled, y_cfr)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        population = int(request.form['population'])

        new_data = pd.DataFrame({'latitude': [latitude], 'longitude': [longitude], 'population': [population]})
        new_data_scaled = scaler.transform(new_data)

        predicted_deaths = rf_deaths.predict(new_data_scaled)[0]
        predicted_cases = rf_cases.predict(new_data_scaled)[0]
        predicted_cfr = rf_cfr.predict(new_data_scaled)[0]

        return jsonify({
            'predicted_deaths': round(predicted_deaths, 2),
            'predicted_cases': round(predicted_cases, 2),
            'predicted_cfr': round(predicted_cfr, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
