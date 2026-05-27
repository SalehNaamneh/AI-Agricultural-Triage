import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from PIL import Image

from crop_config import load_all_crops
from predict import predict_pil

app = FastAPI(title="AgriTriage Image Analyzer", version="2.0.0")

_crops = {}

@app.on_event("startup")
def startup():
    global _crops
    _crops = load_all_crops()
    for crop_id, crop in _crops.items():
        if crop.model_path.exists():
            from predict import get_model
            get_model(crop)
            print(f"Loaded model for crop: {crop_id}")
        else:
            print(f"No model found for crop '{crop_id}' — train first with train.py --crop {crop_id}")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Query(default="onion", description="Crop ID (e.g. onion, tomato)"),
):
    if crop not in _crops:
        available = list(_crops.keys())
        raise HTTPException(status_code=400, detail=f"Unknown crop '{crop}'. Available: {available}")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    try:
        data  = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return predict_pil(image, crop_id=crop)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/crops")
def list_crops():
    return {
        crop_id: {"name_en": c.name_en, "name_he": c.name_he, "icon": c.icon,
                  "num_classes": c.num_classes, "model_ready": c.model_path.exists()}
        for crop_id, c in _crops.items()
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "crops": list(_crops.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=False)
