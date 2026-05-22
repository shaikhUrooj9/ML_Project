import os
import numpy as np
import onnxruntime as ort
from PIL import Image

def get_onnx_session(model_path='model.onnx'):
    """Loads the ONNX runtime inference session model safely."""
    if os.path.exists(model_path):
        return ort.InferenceSession(model_path)
    return None

def preprocess_image(image_path):
    """Resizes, normalizes, and shapes an image for the AI model network."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    
    # Convert pixels to numpy array and scale to [0, 1]
    img_data = np.array(image).astype(np.float32) / 255.0
    
    # Standard ImageNet normalization coefficients
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # Rearrange shape channels from (H, W, C) to (C, H, W) and add batch dimension (1, C, H, W)
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

def load_labels(labels_path='labels.txt'):
    """Reads class names from labels file map."""
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return ["buildings", "forest", "glacier", "mountain", "sea", "street"]