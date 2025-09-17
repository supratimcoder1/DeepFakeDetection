# In preprocessing.py (Final Corrected Version)

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image # Import the Python Imaging Library

IMG_SIZE = 224

# The transforms pipeline is correct and remains the same.
# It expects a PIL Image as input.
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path_or_array):
    """
    Preprocesses an image or a video frame to be identical to the training pipeline.
    """
    if isinstance(image_path_or_array, str):
        # For image files, load with PIL to match the training pipeline exactly.
        img = Image.open(image_path_or_array).convert('RGB')
    
    elif isinstance(image_path_or_array, np.ndarray):
        # For video frames (which are numpy arrays from cv2), first convert
        # BGR to RGB, then convert to a PIL image.
        img = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
    else:
        raise TypeError(f"Unsupported input type: {type(image_path_or_array)}")

    # Apply the transforms
    return transform(img)


def extract_frames(video_path, every_n=5, max_frames=32):
    """
    Extracts frames from a video file using OpenCV.
    This function is correct and remains the same.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames