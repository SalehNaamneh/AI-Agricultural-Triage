import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

from predict import predict_pil, get_model

app = FastAPI(title="Onion Disease Image Analyzer", version="1.0.0")

# Pre-load model on startup
@app.on_event("startup")
def startup():
    get_model()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    try:
        data  = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return predict_pil(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=False)
