import pandas as pd
import statsmodels.formula.api as smf
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load Excel dataset
cars = pd.read_excel("Cars.xlsx")

# Train model
model = smf.ols('MPG ~ VOL + SP + HP', data=cars).fit()

@app.route("/")
def home():
    return "Multilinear Regression API is Live 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({"MPG_Prediction": float(prediction)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
