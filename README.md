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
