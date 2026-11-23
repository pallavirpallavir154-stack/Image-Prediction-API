from fastapi import FastAPI, UploadFile, File, HTTPException
from app.model import predict_image
from app.utils import preprocess_image

app = FastAPI(
    title="AI Image Prediction API",
    description="An AI-powered image classification API using MobileNetV2",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"message": "Welcome to AI Image Prediction API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are allowed")

    try:
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)
        predictions = predict_image(image)

        return {
            "filename": file.filename,
            "predictions": predictions,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))