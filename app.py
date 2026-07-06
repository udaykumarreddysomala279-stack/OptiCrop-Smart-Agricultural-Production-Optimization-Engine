from crop_info import crop_data
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and label encoder
lr = joblib.load("crop_recommendation_model.pkl")
encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        prediction = lr.predict(features)
        crop = encoder.inverse_transform(prediction)
        
        print("Input Features:", features)
        print("Encoded Prediction:", prediction)
        print("Decoded Prediction:", crop)

        crop_name = crop[0].lower()

        image = crop_name + ".jpg"

        info = crop_data.get(crop_name)

        return render_template(
            "result.html",
            prediction=crop_name.title(),
            image=image,
            info=info
        )

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)