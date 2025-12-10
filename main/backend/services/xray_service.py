# backend/services/xray_service.py

import os
from pathlib import Path
import torch
from models.xray_model import load_chexnet_model, predict_xray

# Resolve project root and weight path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHT_PATH = PROJECT_ROOT / 'model_assests' / 'xray' / 'xray.pth.tar'

_xray_model = None

def init_xray_model(weight_path: Path = DEFAULT_WEIGHT_PATH, device: str = 'cpu') -> None:
    """
    Load and cache the CheXNet X-ray model.
    """
    global _xray_model

    weight_path = Path(weight_path)
    if not weight_path.is_file():
        raise FileNotFoundError(f"X-ray weights not found at {weight_path}. Please ensure the model weights are downloaded and placed in the correct directory.")

    _xray_model = load_chexnet_model(str(weight_path), device=device)

# Try initializing on import
try:
    init_xray_model()
except Exception as e:
    print(f"Warning: could not load X-ray model: {e}")


import pydicom
import numpy as np
from PIL import Image
import tempfile

def process_xray(image_path: str, device: str = 'cpu', top_k: int = 3, use_class_thresholds: bool = False) -> list:
    if _xray_model is None:
        raise RuntimeError("X-ray model not initialized. Call init_xray_model() first.")

    ext = os.path.splitext(image_path)[1].lower()
    allowed_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.dcm']
    if ext not in allowed_exts:
        raise ValueError(f"Unsupported file type: {ext}")

    # Handle DICOM files
    if ext == '.dcm':
        try:
            ds = pydicom.dcmread(image_path)
            pixel_array = ds.pixel_array
            
            # Normalize to 0-255
            if pixel_array.max() > 0:
                pixel_array = pixel_array / pixel_array.max() * 255.0
            
            im = Image.fromarray(pixel_array.astype(np.uint8))
            if im.mode != 'RGB':
                im = im.convert('RGB')
            
            # Save as temp PNG
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_png_path = tmp.name
            
            im.save(temp_png_path)
            
            # Predict using the temp PNG
            try:
                results = predict_xray(_xray_model, temp_png_path, top_k=top_k, device=device, use_class_thresholds=use_class_thresholds)
            finally:
                # Cleanup temp file
                if os.path.exists(temp_png_path):
                    os.remove(temp_png_path)
            
            return results
            
        except Exception as e:
            print(f"Error processing DICOM: {e}")
            raise ValueError(f"Failed to process DICOM file: {e}")

    return predict_xray(_xray_model, image_path, top_k=top_k, device=device, use_class_thresholds=use_class_thresholds)

