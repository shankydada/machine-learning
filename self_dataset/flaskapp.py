import numpy as np
from flask import Flask,request,render_template
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("./model.pkl", "rb"))

@flask_app.route('/')
def Home():
    return render_template("index.html")
@flask_app.route("/predict",methods=["POST"])
def predict():
    float_features=[float(x) for c in request.form.values()]
    features =[np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html",predicition_text="The Predicted value is {} ".format(prediction))

if __name__ == "--main--":
    flask_app.run(debug=True)
