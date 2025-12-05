import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json

class VegetableModel:
    def __init__(self, model_path="best_model_ema.pth", label_path="labels.json"):
        with open(label_path, "r") as f:
            self.labels = json.load(f)

        self.model = torch.load(model_path, map_location="cpu")
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image):
        img = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(img)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, dim=1)[0][predicted].item()
        return self.labels[str(predicted.item())], float(confidence)
