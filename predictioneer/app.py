from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load('Case_Fatality_Ratio.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form input data
    lat = float(request.form['Lat'])
    long_ = float(request.form['Long_'])
    deaths = int(request.form['Deaths'])
    
    # Prepare the input data for prediction
    new_data = pd.DataFrame({
        'Lat': [lat],
        'Long_': [long_],
        'Deaths': [deaths]
    })
    
    # Predict using the trained model
    prediction = model.predict(new_data)[0]
    
    return render_template('index.html', prediction=prediction, lat=lat, long_=long_, deaths=deaths)

if __name__ == '__main__':
    app.run(debug=True)
