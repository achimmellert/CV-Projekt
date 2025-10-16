import json
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import io
import timm

app = FastAPI(title="Emotion Detection API")

# Erlaube CORS für deine Domain (korrigiere den Leerzeichen-Fehler!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://achimmellert.de"],  # ← Entferne die überflüssigen Leerzeichen!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# 1. Modell & Label-Mapping laden
# ===========================

# Lade Labels (sollte aus dem Training stammen)
with open("class_labels.json", "r") as jf:
    label_to_idx = json.load(jf)
    idx_to_class = {int(v): k for k, v in label_to_idx.items()}

# Lade das EfficientNet-B0-Modell mit 7 Klassen
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=7)
model.load_state_dict(torch.load("best_efficientnet_b0_overall.pth", map_location="cpu", weights_only=True))
model.eval()
print("✅ EfficientNet-B0 Modell erfolgreich geladen.")

# ===========================
# 2. Transform (WICHTIG: entspricht dem Training!)
# ===========================

# Diese Normalisierung muss EXAKT der vom Training entsprechen!
# FastAI hat Normalize.from_stats(*imagenet_stats) verwendet → das ist:
transform = T.Compose([
    T.Resize((224, 224)),                 # ↑ Wie im Training!
    T.ToTensor(),                         # Wandelt PIL in [0,1] Tensor um
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet Mean
                std=[0.229, 0.224, 0.225])   # ImageNet Std
])

# ===========================
# 3. Funktion: Gesicht extrahieren mit MediaPipe
# ===========================

def extract_first_face(pil_image):
    mp_face_detection = mp.solutions.face_detection
    np_image = np.array(pil_image)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as fd:
        results = fd.process(np_image)
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

# ===========================
# 4. Endpoint: Startseite ausliefern
# ===========================

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

# ===========================
# 5. Endpoint: Emotion vorhersagen
# ===========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 1. Gesicht extrahieren
    face_pil = extract_first_face(pil_image)
    if face_pil is None:
        return {"error": "No face detected", "success": False}

    # 2. Transformieren (224x224 RGB, ImageNet-Norm)
    input_tensor = transform(face_pil).unsqueeze(0)  # [C,H,W] → [1,C,H,W]

    # 3. Inferenz
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