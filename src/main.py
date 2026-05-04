import json
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from src.models.cnn import CNN


app = FastAPI(title="Emotion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Für Produktion auf Frontend-Domain anpassen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Lade Modell
model = CNN(num_classes=7, dropout_b=0.2, dropout_fc=0.4)
model.load_state_dict(torch.load("models/best_simple_cnn_model.pth", map_location="cpu"))
model.eval()

# 2. Lade Labels
with open("data/class_labels.json", "r") as jf:
    label_to_idx = json.load(jf)
    idx_to_class = {int(v): k for k, v in label_to_idx.items()}

# 3. Transform
transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((48, 48)),
    T.ToTensor(),
    T.Normalize(mean=[0.5077], std=[0.2551])
])

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)


def extract_first_face(pil_image):
    np_image = np.array(pil_image)

    # Nutze den globalen Detektor
    results = face_detector.process(np_image)
    if not results.detections:
        return None

    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box

    h, w = np_image.shape[:2]
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    width = int(bbox.width * w)
    height = int(bbox.height * h)
    x, y = max(0, x), max(0, y)
    width, height = min(width, w - x), min(height, h - y)

    face_pil = pil_image.crop((x, y, x + width, y + height))

    if face_pil.size[0] == 0 or face_pil.size[1] == 0:
        return None

    return face_pil


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Emotion Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    face_pil = extract_first_face(pil_image)
    if face_pil is None:
        return {"error": "No face detected"}

    input_tensor = transform(face_pil).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, class_idx = torch.max(probs, dim=1)
        emotion = idx_to_class[class_idx.item()]

        return {
            "emotion": emotion,
            "confidence": confidence.item(),
            "success": True
        }
