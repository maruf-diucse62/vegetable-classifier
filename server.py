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
DRIVE_FILE_ID = "YOUR_FILE_ID"  # Replace with your actual Google Drive file ID
MODEL_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

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
    
    image_file = request.files["image"]
    result = model.predict(image_file)
    return jsonify(result)

# --- Run app ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
