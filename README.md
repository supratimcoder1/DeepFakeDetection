# 🎭 Deepfake Video & Image Detection

An end-to-end **deepfake detection system** that can analyze **videos** or **images** and predict whether the content is **REAL** or **FAKE**, along with a confidence score.

This project uses a **hybrid deep learning ensemble** (EfficientNet-B4 + ConvNeXt-Base + Swin-Small) with an **FFT branch** for frequency-domain analysis, followed by a **logistic regression meta-model** for optimal decision-making.  

The backend is built with **FastAPI**, and a simple **HTML+CSS+JS frontend** allows users to upload files and view results in real time.

---

## 📑 Table of Contents
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup Instructions](#-setup-instructions)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Sample Predictions](#-sample-predictions)
- [Future Work](#-future-work)
- [License](#-license)

---

## ✨ Features
✅ **Deepfake Video Detection** – Extracts frames from videos, performs per-frame classification, and aggregates results for final decision.  
✅ **Image Detection Support** – Classifies single images as REAL/FAKE.  
✅ **Ensemble Learning** – Combines EfficientNet-B4, ConvNeXt-Base, and Swin-Small predictions using a logistic regression meta-model for higher accuracy.  
✅ **FFT-Based Feature Fusion** – Captures frequency artifacts often present in deepfakes.  
✅ **Confidence Score** – Returns a probability-based confidence value (0–1).  
✅ **REST API** – Exposes `/predict/image` and `/predict/video` endpoints for easy integration.  
✅ **Frontend** – Simple web interface for uploading and visualizing predictions.

---

## 🛠 Tech Stack

| Layer             | Technology Used |
|------------------|----------------|
| **Model Training** | PyTorch, timm, sklearn |
| **Backend**       | FastAPI, Uvicorn |
| **Frontend**      | HTML, CSS, Vanilla JS |
| **Model Serving** | TorchScript + joblib |
| **Other Tools**   | OpenCV (frame extraction), NumPy, Pandas |

---

## 📂 Project Structure

```bash
deepfake-detection/
│
├── backend/
│   ├── app.py                 # FastAPI app (routes & API)
│   ├── inference.py           # Model loading, preprocessing & predictions
│   └── ensemble_models/       # Trained weights + meta-model
│
├── frontend/
│   ├── index.html             # Landing page
│   └── detection.html         # Upload page (video/image detection UI)
│
|
├── requirements.txt           # Python dependencies
└── README.md                  # This file

⚙️ Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/<your-username>/deepfake-detection.git
cd deepfake-detection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Place Model Files

Download and place the following files inside backend/ensemble_models/:

model_efficientnet_b4_best.pth

model_convnext_base_best.pth

model_swin_small_patch4_window7_224_best.pth

meta_logreg.pkl

meta_scaler.pkl

4️⃣ Run the Backend
cd backend
uvicorn app:app --reload


Backend will be available at http://127.0.0.1:8000.

5️⃣ Open Frontend

Open frontend/index.html in your browser and start uploading files.

🖥 Usage
🟢 Health Check
GET /health


Response:

{ "status": "ok" }

🟠 Predict on Image
POST /predict/image
Content-Type: multipart/form-data
file=@sample_image.jpg

🔵 Predict on Video
POST /predict/video
Content-Type: multipart/form-data
file=@sample_video.mp4


Response:

{
  "file_type": "video",
  "frames_processed": 28,
  "confidence": 0.943,
  "label": "FAKE"
}

📊 Sample Predictions
File	Type	Confidence	Prediction
real_video.mp4	Video	0.11	✅ REAL
deepfake.mp4	Video	0.92	❌ FAKE
image1.jpg	Image	0.27	✅ REAL
deepfake_face.png	Image	0.88	❌ FAKE
🚀 Future Work

🔬 Frame-Level Visualization – Heatmaps for which frames contribute most to the FAKE classification.

🌐 Deploy on Cloud – Dockerize backend and deploy on AWS/GCP.

📱 Mobile-Friendly Frontend – Improve UI/UX for smartphone uploads.

🧠 Support for Audio Deepfake Detection – Extend to detect manipulated audio tracks.

📜 License

This project is licensed under the MIT License. See LICENSE for details.