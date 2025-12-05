from flask import Flask, request, jsonify
from model_loader import VegetableModel
from PIL import Image
import flask_cors

app = Flask(__name__)
flask_cors.CORS(app)

model = VegetableModel("best_model_ema.pth", "labels.json")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(request.files["image"]).convert("RGB")
    label, confidence = model.predict(img)

    return jsonify({
        "label": label,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
