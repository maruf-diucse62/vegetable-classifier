# server.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import gdown
from model_loader import VegetableModel

app = Flask(__name__)
CORS(app)

# --- Model setup ---
MODEL_PATH = "best_model_ema.pth"

# Google Drive file ID for your model
DRIVE_FILE_ID = "18plKhXwLZQCeit5yW6aEQ2XdCCoKs1xD"
MODEL_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Download completed.")

# Load model
model = VegetableModel(MODEL_PATH, "labels.json")
print("Model loaded successfully!")

# --- Routes ---
@app.route("/")
def home():
    return "Vegetable Classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files["image"]
    label, confidence = model.predict(img_file)
    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
