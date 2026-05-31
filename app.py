from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io
import base64
from datetime import datetime
from pymongo import MongoClient
import os

app = FastAPI()

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["drowsiness_detection_db"]
collection = db["prediction_results"]

MODEL = None  # lazy loading (IMPORTANT FIX)


def load_model():
    global MODEL
    if MODEL is None:
        from tensorflow.keras.models import load_model
        MODEL = load_model("ultimate_drowsiness_system.h5")
    return MODEL


@app.get("/")
def home():
    return {"message": "API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    model = load_model()

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    score = float(model.predict(img, verbose=0)[0][0])

    if score < 0.5:
        prediction = "Drowsy"
        confidence = (1 - score) * 100
    else:
        prediction = "Non Drowsy"
        confidence = score * 100

    collection.insert_one({
        "timestamp": datetime.utcnow(),
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "image_base64": base64.b64encode(image_bytes).decode()
    })

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }
