"""FastAPI wrapper around the best trained brain tumor classifier.

Run:  uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
MAX_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    import tensorflow as tf  # imported here so startup errors are clear

    model_path = MODEL_DIR / "best_model.keras"
    meta_path = MODEL_DIR / "model_meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise RuntimeError(f"Model files not found in {MODEL_DIR}. Run train.py first.")
    state["model"] = tf.keras.models.load_model(model_path)
    state["meta"] = json.loads(meta_path.read_text())
    yield
    state.clear()


app = FastAPI(
    title="Brain Tumor Detection API",
    description="Binary MRI classifier (tumor / no tumor). Research use only, not a medical device.",
    version="1.0.0",
    lifespan=lifespan,
)


class Prediction(BaseModel):
    label: str
    tumor_probability: float
    confidence: float
    threshold: float
    model_name: str
    disclaimer: str = "For research/educational use only. Not a medical diagnosis."


def preprocess(raw: bytes, size: int) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((size, size))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not read the file as an image.")
    # Raw 0-255 pixels: rescaling/preprocess_input lives inside the saved model.
    return np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state,
            "model_name": state.get("meta", {}).get("model_name")}


@app.get("/model-info")
def model_info():
    meta = state["meta"]
    return {k: meta[k] for k in ("model_name", "img_size", "threshold", "test_auc", "val_auc")}


@app.post("/predict", response_model=Prediction)
def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPG or PNG image.")
    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File larger than 10 MB.")

    meta = state["meta"]
    x = preprocess(raw, meta["img_size"])
    prob = float(state["model"].predict(x, verbose=0)[0][0])
    is_tumor = prob >= meta["threshold"]
    return Prediction(
        label=meta["labels"]["1" if is_tumor else "0"],
        tumor_probability=round(prob, 4),
        confidence=round(prob if is_tumor else 1 - prob, 4),
        threshold=meta["threshold"],
        model_name=meta["model_name"],
    )
