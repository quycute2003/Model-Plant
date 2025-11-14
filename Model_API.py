from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
from PIL import Image
import uvicorn
import os

# ======================================
# CONFIG
# ======================================
IMG_SIZE = (224, 224)

DISEASE_MODEL_PATH = "disease.keras"
DISEASE_LABELS_PATH = "disease_labels.txt"

PLANT_MODEL_PATH = "plant.keras"
PLANT_LABELS_PATH = "plant_labels.txt"

# ======================================
# LOAD MODELS
# ======================================
print("🔄 Loading models...")

model_disease = tf.keras.models.load_model(DISEASE_MODEL_PATH)
print("✔ Loaded disease model")

model_plant = tf.keras.models.load_model(PLANT_MODEL_PATH)
print("✔ Loaded plant model")

# ======================================
# LOAD LABELS
# ======================================
with open(DISEASE_LABELS_PATH, "r", encoding="utf-8") as f:
    disease_labels = [x.strip() for x in f.readlines()]

with open(PLANT_LABELS_PATH, "r", encoding="utf-8") as f:
    plant_labels = [x.strip() for x in f.readlines()]

print("✔ Loaded label files")

# EfficientNetV2 preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# ======================================
# IMAGE PREPROCESS FUNCTION
# ======================================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)  # (1,224,224,3)
    return arr

# ======================================
# CREATE FASTAPI
# ======================================
app = FastAPI(title="Plant & Disease Classification API")

# ==============================================================
# 1️⃣ API — Predict Disease
# ==============================================================
@app.post("/predict_disease")
async def predict_disease(file: UploadFile = File(...)):

    img = Image.open(file.file)
    arr = preprocess_image(img)

    preds = model_disease.predict(arr)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "task": "leaf disease detection",
        "class": disease_labels[class_id],
        "confidence": round(confidence, 4)
    }

# ==============================================================
# 2️⃣ API — Predict Plant Species
# ==============================================================
@app.post("/predict_plant")
async def predict_plant(file: UploadFile = File(...)):

    img = Image.open(file.file)
    arr = preprocess_image(img)

    preds = model_plant.predict(arr)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "task": "plant species classification",
        "class": plant_labels[class_id],
        "confidence": round(confidence, 4)
    }

# ======================================
# START SERVER (for local run)
# ======================================
if __name__ == "__main__":
    uvicorn.run("Model_API:app", host="0.0.0.0", port=8000, reload=True)
