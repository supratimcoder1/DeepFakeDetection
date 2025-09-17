import torch
import torch.nn as nn
import timm
import numpy as np
import joblib
from collections import OrderedDict

# Import the authoritative preprocessing functions from your dedicated module
from preprocessing import preprocess_image, extract_frames

# --- Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FFT_EMBED_DIM = 128
DROPOUT = 0.5


# --- Model Class Definitions (These are correct and remain unchanged) ---
class FFTHead(nn.Module):
    def __init__(self, out_dim=FFT_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x_gray = x.mean(dim=1, keepdim=True)
        xf = torch.fft.fft2(x_gray, dim=(-2, -1))
        mag = torch.log1p(torch.abs(xf))
        mag = (mag - mag.mean(dim=(-2, -1), keepdim=True)) / (mag.std(dim=(-2, -1), keepdim=True) + 1e-6)
        return self.net(mag)

class BackboneWithFFT(nn.Module):
    def __init__(self, timm_name, pretrained=False, fft_dim=FFT_EMBED_DIM, dropout=DROPOUT):
        super().__init__()
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.fft_head = FFTHead(out_dim=fft_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim + fft_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        feat = self.backbone(x)
        fft_feat = self.fft_head(x)
        combined = torch.cat([feat, fft_feat], dim=1)
        return self.classifier(combined)


# --- Model Loading and Prediction Logic ---
def load_models():
    """
    Loads all deep learning models, the meta-model, and the feature scaler.
    """
    models = {}
    print("Loading deep learning models...")
    for name, ckpt_path, timm_name in [
        ("effnet", "ensemble_models/model_efficientnet_b4_best.pth", "efficientnet_b4"),
        ("convnext", "ensemble_models/model_convnext_base_best.pth", "convnext_base"),
        ("swin", "ensemble_models/model_swin_small_patch4_window7_224_best.pth", "swin_small_patch4_window7_224")
    ]:
        model = BackboneWithFFT(timm_name)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        models[name] = model.to(DEVICE).eval()

    print("Loading meta-model and feature scaler...")
    models['meta'] = joblib.load("ensemble_models/meta_logreg.pkl")
    models['meta_scaler'] = joblib.load("ensemble_models/meta_scaler.pkl")
    print("All models and scaler loaded successfully.")
    return models

def predict_image(image_path, models):
    """
    Runs prediction on a single image file.
    """
    # Use the imported preprocess_image function
    img_tensor = preprocess_image(image_path).to(DEVICE)
    
    base_preds = []
    for model_name in ['effnet', 'convnext', 'swin']:
        with torch.no_grad():
            logits = models[model_name](img_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            base_preds.append(probs[0, 1].cpu().numpy())
            
    features = np.array(base_preds).reshape(1, -1)
    scaled_features = models['meta_scaler'].transform(features)
    final_prob = models['meta'].predict_proba(scaled_features)[0, 1]
    
    return {
        "file_type": "image",
        "confidence": float(final_prob),
        "label": "FAKE" if final_prob >= 0.5 else "REAL"
    }

def predict_video(video_path, models):
    """
    Runs prediction on a video file by analyzing its frames.
    """
    # Use the imported extract_frames function
    frames = extract_frames(video_path, every_n=5, max_frames=32)
    if not frames:
        return {"error": "No frames could be extracted from the video."}
        
    frame_scores = []
    for frame in frames:
        # Use the imported preprocess_image function for each frame
        img_tensor = preprocess_image(frame).to(DEVICE)
        base_preds = []
        for model_name in ['effnet', 'convnext', 'swin']:
            with torch.no_grad():
                logits = models[model_name](img_tensor.unsqueeze(0))
                probs = torch.softmax(logits, dim=1)
                base_preds.append(probs[0, 1].cpu().numpy())
                
        features = np.array(base_preds).reshape(1, -1)
        scaled_features = models['meta_scaler'].transform(features)
        frame_scores.append(models['meta'].predict_proba(scaled_features)[0, 1])
        
    avg_score = np.mean(frame_scores)
    
    return {
        "file_type": "video",
        "confidence": float(avg_score),
        "label": "FAKE" if avg_score >= 0.5 else "REAL"
    }