from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64
from pymongo import MongoClient
from datetime import datetime

app = FastAPI()

# Load model
model = load_model("ultimate_drowsiness_system.h5")

# MongoDB (replace with your Atlas URI later in Render)
MONGO_URI = "YOUR_MONGO_URI"
client = MongoClient(MONGO_URI)

db = client["drowsiness_detection_db"]
collection = db["prediction_results"]

MODEL_ACCURACY = 94.0


@app.get("/")
def home():
    return {"message": "Drowsiness API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

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

    document = {
        "timestamp": datetime.utcnow(),
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "image_base64": base64.b64encode(image_bytes).decode(),
        "model_accuracy": MODEL_ACCURACY
    }

    collection.insert_one(document)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "model_accuracy": MODEL_ACCURACY
    }
