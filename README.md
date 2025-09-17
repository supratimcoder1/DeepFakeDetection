# Deepfake Video Detection System

This project is a **Deepfake Video Detection System** built with **FastAPI** for the backend and a lightweight **HTML/CSS/JS frontend**.  
It uses a **hybrid ensemble of deep learning models** (EfficientNet-B4, ConvNeXt-Base, Swin-Small) trained with aggressive augmentations, combined with a **logistic regression meta-model** for optimal performance.

---

## 🚀 Features

- **Deepfake Video Detection:** Predicts whether a video is REAL or FAKE with a confidence score.
- **Image Support:** Can also predict on single images.
- **Ensemble Model:** EfficientNet-B4 + ConvNeXt-Base + Swin-Small with FFT features.
- **Meta-Model:** Logistic Regression meta-learner for better decision boundaries.
- **REST API:** Built with FastAPI, fully documented and easy to integrate.
- **Frontend:** Simple HTML interface for file upload and result display.

---

## 📂 Project Structure

```
project-root/
├── backend/
│   ├── app.py                # FastAPI application
│   ├── inference.py          # Model loading & prediction logic
│   ├── preprocessing.py      # Data preprocessing
│   ├── utils.py              # Utility functions
│   └── ensemble_models/      # Saved PyTorch models + meta-model
│
├── frontend/
│   ├── index.html            # Homepage
│   └── detection.html        # Upload & result page
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/deepfake-video-detection.git
cd deepfake-video-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the backend**
```bash
uvicorn backend.app:app --reload
```

4. **Access the API docs**
- Visit: `http://127.0.0.1:8000/docs`

5. **Open the frontend**
- Open `frontend/index.html` in a browser.

---

## 🔧 API Endpoints

| Method | Endpoint         | Description |
|-------|-----------------|-------------|
| GET   | `/health`        | Health check for the API |
| POST  | `/predict/image` | Upload an image and get prediction |
| POST  | `/predict/video` | Upload a video and get prediction |

### Example cURL Request:
```bash
curl -X POST "http://127.0.0.1:8000/predict/video"   -F "file=@sample_video.mp4"
```

---

## 📊 Model Details

- **EfficientNet-B4** – Lightweight and efficient CNN backbone.
- **ConvNeXt-Base** – Modern convolutional architecture.
- **Swin-Small** – Transformer-based architecture for global attention.
- **FFT Head** – Learns frequency-domain artifacts common in deepfakes.
- **Meta-Model** – Logistic Regression trained on validation set predictions.

Validation results:  
- **Val AUC:** ~0.964  
- **Test Video AUC:** ~0.938

---

## 🖼 Frontend Demo

The frontend consists of two pages:
1. **index.html** – Landing page with navigation.
2. **detection.html** – File upload & result display.

---

## 🧠 Future Improvements

- Support for live video streams.
- Model quantization for faster inference.
- Deployable Docker image for production use.

---