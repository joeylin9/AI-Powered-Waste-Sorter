# app.py - Flask backend for serving PyTorch model
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class WasteSorterModel:
    def __init__(self, model_path="model.pt", labels_path="labels.txt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = self.load_labels(labels_path)
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        
    def load_labels(self, labels_path):
        """Load class labels from file"""
        try:
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(labels)} class labels")
            return labels
        except FileNotFoundError:
            print(f"Labels file {labels_path} not found!")
            return []
    
    def load_model(self, model_path):
        """Load the trained PyTorch model"""
        try:
            # Create model architecture (same as training)
            model = models.resnet50(weights=None)  # Don't load pretrained weights
            
            # Recreate the classifier head (same as training)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, len(self.labels))
            )
            
            # Load the saved state dict
            if model_path.endswith('model.pt'):
                # If it's a checkpoint with additional info
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                # If it's just the state dict
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            model.to(self.device)
            model.eval()
            print(f"Model loaded successfully on {self.device}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def get_transform(self):
        """Get the same transform used during validation"""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image, return_all=False):
        """Make prediction on image"""
        if self.model is None:
            return None, 0.0, None
        
        try:
            # Transform image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get top prediction
                confidence, predicted_idx = torch.max(probabilities, 0)
                predicted_class = self.labels[predicted_idx.item()]
                
                # Get all predictions if requested
                all_predictions = None
                if return_all:
                    all_predictions = {}
                    for i, label in enumerate(self.labels):
                        all_predictions[label] = probabilities[i].item()
                
                return predicted_class, confidence.item(), all_predictions
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, None

# Initialize model
waste_model = WasteSorterModel()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Check if all predictions are requested
        return_all_predictions = data.get('return_all_predictions', False)
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/png;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Make prediction
        predicted_class, confidence, all_predictions = waste_model.predict(image, return_all=return_all_predictions)
        
        if predicted_class is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        response = {
            'className': predicted_class,
            'probability': confidence,
            'success': True
        }
        
        # Add all predictions if requested
        if return_all_predictions and all_predictions:
            response['all_predictions'] = all_predictions
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': waste_model.model is not None,
        'device': str(waste_model.device),
        'num_classes': len(waste_model.labels)
    })

@app.route('/save-image', methods=['POST'])
def save_image():
    import os, base64
    from datetime import datetime

    data = request.get_json()
    image_data = data.get('image')
    class_name = data.get('className')

    if not image_data or not class_name:
        return jsonify({"success": False, "error": "Missing data"}), 400

    base_dir = "dataset"
    class_dir = os.path.join(base_dir, class_name.upper())
    os.makedirs(class_dir, exist_ok=True)

    counter = 1
    while True:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{counter}.png"
        filepath = os.path.join(class_dir, filename)
        if not os.path.exists(filepath):
            break
        counter += 1

    try:
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(image_data))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({"success": True, "filename": filename})

@app.route('/retrain', methods=['POST'])
def retrain():
    import threading

    def background_train():
        from train_model import train_from_data
        train_from_data()
        # Reload the model after training:
        waste_model.model = waste_model.load_model("model.pt")
        waste_model.labels = waste_model.load_labels("labels.txt")

    thread = threading.Thread(target=background_train)
    thread.start()
    return jsonify({"success": True, "message": "Retraining started in background"})

if __name__ == '__main__':
    print("Starting Waste Sorter API...")
    print(f"Model loaded: {waste_model.model is not None}")
    print(f"Number of classes: {len(waste_model.labels)}")
    app.run(debug=True, host='0.0.0.0', port=5000)