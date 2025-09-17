import os
import shutil
import tempfile
import time
import logging
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from inference import load_models, predict_video, predict_image

# ==========================================================
# Configure Logging
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deepfake_api")

# ==========================================================
# FastAPI App
# ==========================================================
app = FastAPI(title="Deepfake Detection API", version="1.0")

# Enable CORS (allow frontend to talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Loading models into memory...")
    app.state.models = load_models()
    logger.info("Models successfully loaded and ready.")

@app.get("/health")
async def health():
    logger.info("Health check endpoint hit.")
    return {"status": "ok"}

@app.post("/predict/image")
async def predict_image_endpoint(file: UploadFile = File(...), request: Request = None):
    start_time = time.time()
    logger.info(f"Received IMAGE prediction request: filename={file.filename}, client={request.client.host if request else 'unknown'}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_image(file_path, app.state.models)

    elapsed = time.time() - start_time
    logger.info(f"IMAGE prediction completed: label={result['label']}, confidence={result['confidence']:.4f}, time={elapsed:.2f}s")
    return JSONResponse(content=result)

@app.post("/predict/video")
async def predict_video_endpoint(file: UploadFile = File(...), request: Request = None):
    start_time = time.time()
    logger.info(f"Received VIDEO prediction request: filename={file.filename}, client={request.client.host if request else 'unknown'}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_video(file_path, app.state.models)

    elapsed = time.time() - start_time
    logger.info(f"VIDEO prediction completed: label={result['label']}, confidence={result['confidence']:.4f}, time={elapsed:.2f}s")
    return JSONResponse(content=result)
