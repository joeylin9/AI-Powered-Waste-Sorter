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
import os, base64
from datetime import datetime
import threading
from threading import Lock
import time

app = Flask(__name__)

# buffer and lock
pending_images = []
pending_lock   = Lock()

# Add training lock to prevent model reloading during training
training_lock = Lock()
is_training = False

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class WasteSorterModel:
    def __init__(self, model_path="model_folder/model.pt", labels_path="model_folder/labels.txt"):
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
            if model_path.endswith('.pt'):
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
            print(f"Model loaded successfully from {model_path} on {self.device}")
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
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
    global is_training
    return jsonify({
        'status': 'healthy',
        'model_loaded': waste_model.model is not None,
        'device': str(waste_model.device),
        'num_classes': len(waste_model.labels),
        'is_training': is_training
    })

@app.route('/save-image', methods=['POST'])
def save_image():
    global is_training
    data = request.get_json()
    image_data = data.get('image')
    class_name = data.get('className')
    train_model_flag = False

    # 1) Validate input
    if not image_data or not class_name:
        return jsonify({"success": False, "error": "Missing data"}), 400

    # 2) Buffer incoming images
    with pending_lock:
        pending_images.append({
            "image":      image_data,
            "class_name": class_name.upper()
        })
        if len(pending_images) >= 3:
            to_save = pending_images.copy()
            pending_images.clear()
            train_model_flag = True
        else:
            to_save = []

    # 3) Persist buffered images to disk
    saved_files = []
    for entry in to_save:
        cls_dir = os.path.join("dataset", entry["class_name"])
        os.makedirs(cls_dir, exist_ok=True)

        # find a non‐colliding filename
        counter = 1
        while True:
            fn = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{counter}.png"
            path = os.path.join(cls_dir, fn)
            if not os.path.exists(path):
                break
            counter += 1

        try:
            with open(path, "wb") as f:
                f.write(base64.b64decode(entry["image"]))
            saved_files.append(fn)
        except Exception as e:
            app.logger.error(f"Failed to write {fn}: {e}")

    # 4) If we hit the batch size, only start training if none is running
    start_thread = False
    if train_model_flag:
        with training_lock:
            if not is_training:
                is_training = True
                start_thread = True

    if start_thread:
        thread = threading.Thread(target=background_train)
        thread.daemon = True
        thread.start()
        return jsonify({
            "success": True,
            "message": "Retraining started in background with model versioning",
            "saved_files": saved_files
        })

    # 5) Otherwise, just acknowledge the save
    return jsonify({
        "success": True,
        "message": "Image saved, waiting for more images to retrain.",
        "saved_files": saved_files
    })

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """Manual endpoint to reload the model after training"""
    global waste_model, is_training
    
    if is_training:
        return jsonify({"success": False, "message": "Training in progress"}), 503
    
    try:
        # Reload labels and model
        waste_model.labels = waste_model.load_labels("model_folder/labels.txt")
        waste_model.model = waste_model.load_model("model_folder/model.pt")
        
        if waste_model.model is not None:
            return jsonify({"success": True, "message": "Model reloaded successfully"})
        else:
            return jsonify({"success": False, "message": "Failed to load model"})
            
    except Exception as e:
        return jsonify({"success": False, "message": f"Error reloading model: {e}"})

def background_train():
    """Run retraining in the background, with versioning and atomic swap."""
    global is_training
    try:
        print("Starting background training with model versioning...")
        from train_model import train_from_data
        import shutil

        # Backup existing model
        if os.path.exists("model_folder/model.pt"):
            shutil.copy2("model_folder/model.pt", "model_folder/model_backup.pt")
            print("Backed up model.pt → model_backup.pt")

        # Train new model
        print("Training new model → model_new.pt")
        train_from_data(output_model_path="model_folder/model_new.pt")

        # Give filesystem time, then validate
        time.sleep(2)
        if os.path.exists("model_folder/model_new.pt") and os.path.exists("model_folder/labels.txt"):
            try:
                torch.load("model_folder/model_new.pt", map_location="cpu")
                print("New model validated")

                # Swap files
                if os.path.exists("model_folder/model.pt"):
                    os.remove("model_folder/model.pt")
                os.rename("model_folder/model_new.pt", "model_folder/model.pt")
                print("Model updated atomically")

                # Cleanup backup
                os.remove("model_folder/model_backup.pt")
            except Exception as e:
                print(f"Validation failed: {e}")
                # restore
                if os.path.exists("model_folder/model_backup.pt"):
                    if os.path.exists("model_folder/model_new.pt"):
                        os.remove("model_folder/model_new.pt")
                    shutil.copy2("model_folder/model_backup.pt", "model_folder/model.pt")
                    print("Restored backup model")
        else:
            print("Training output missing; skipping swap")

    except Exception as e:
        print(f"background_train error: {e}")
        # on error, restore if needed
        if os.path.exists("model_folder/model_backup.pt") and not os.path.exists("model_folder/model.pt"):
            shutil.copy2("model_folder/model_backup.pt", "model_folder/model.pt")
            print("Restored backup after error")
    finally:
        with training_lock:
            is_training = False
            print("background_train: is_training cleared")

if __name__ == '__main__':
    print("Starting Waste Sorter API...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)