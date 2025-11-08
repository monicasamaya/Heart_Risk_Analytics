# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS  
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  #  allow requests from any origin 

# Load the trained Gradient Boosting pipeline
model = joblib.load("gb_model.pkl")
print(" Model loaded successfully.")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  
        df = pd.DataFrame([data]) 
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1][0]
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

