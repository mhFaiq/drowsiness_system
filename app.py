from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64
from datetime import datetime
from pymongo import MongoClient
import os

app = FastAPI()

# ---------------- CORS (MUST BE HERE) ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MONGO ----------------
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["drowsiness_detection_db"]
collection = db["predictions"]

# ---------------- MODEL (LOAD ONCE) ----------------
model = load_model("ultimate_drowsiness_system.h5")


@app.get("/")
def home():
    return {"message": "API Running"}


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

    # SAVE TO MONGO (SAFE)
    try:
        collection.insert_one({
            "timestamp": datetime.utcnow(),
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "image": base64.b64encode(image_bytes).decode()
        })
    except Exception as e:
        print("Mongo error:", e)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }
