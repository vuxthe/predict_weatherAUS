import pickle

import numpy as np
from flask import Flask, request, render_template, url_for
from markupsafe import escape

app = Flask(__name__)

# load the pickle model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    # print(float_features)
    features = [np.array(float_features)]
    # predict_value = float_features[2]
    predict_value = model.predict(features)
    if (predict_value==0):
        predict_value = "Sun"
    else: predict_value = "Rain"

    return render_template("index.html", predict_text = f"The weather tomorrow is {predict_value}")

if __name__ == '__main__':
    app.run()