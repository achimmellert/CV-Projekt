import json
import mediapipe as mp
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


# ===========================
# 1. Modell & Label-Mapping laden (nur einmal beim Start)
# ===========================

@st.cache_resource
def load_model_and_labels():

    class CNN(nn.Module):
        def __init__(self, num_classes=7, dropout_b=0.2, dropout_fc=0.4):
            super().__init__()

            # Block 1
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.dropout1 = nn.Dropout(p=dropout_b)

            # Block 2
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(128)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.dropout2 = nn.Dropout(p=dropout_b)

            # Block 3
            self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm2d(256)
            self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.bn6 = nn.BatchNorm2d(256)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.dropout3 = nn.Dropout(p=dropout_b)

            # Flatten
            self.flatten = nn.Flatten()

            # Fully Connected
            self.fc1 = nn.Linear(256 * 6 * 6, 512)  # ()
            self.bn7 = nn.BatchNorm1d(512)
            self.dropout4 = nn.Dropout(dropout_fc)

            self.fc2 = nn.Linear(512, 256)
            self.bn8 = nn.BatchNorm1d(256)
            self.dropout5 = nn.Dropout(dropout_fc)

            self.fc3 = nn.Linear(256, num_classes)


        def forward(self, x):

            # Block 1
            x = F.leaky_relu(self.bn1(self.conv1(x)))
            x = F.leaky_relu(self.bn2(self.conv2(x)))
            x = self.pool1(x)
            x = self.dropout1(x)

            # Block 2
            x = F.leaky_relu(self.bn3(self.conv3(x)))
            x = F.leaky_relu(self.bn4(self.conv4(x)))
            x = self.pool2(x)
            x = self.dropout2(x)

            # Block 3
            x = F.leaky_relu(self.bn5(self.conv5(x)))
            x = F.leaky_relu(self.bn6(self.conv6(x)))
            x = self.pool3(x)
            x = self.dropout3(x)

            # Flatten
            x = self.flatten(x)

            # Fully Connected
            x = F.leaky_relu(self.bn7(self.fc1(x)))
            x = self.dropout4(x)

            x = F.leaky_relu(self.bn8(self.fc2(x)))
            x = self.dropout5(x)

            x = self.fc3(x)

            return x


    model = CNN(num_classes=7, dropout_b=0.2, dropout_fc=0.4)

    model.load_state_dict(torch.load("best_simple_cnn_model.pth", map_location="cpu"))

    model.eval()

    with open("class_labels.json", "r") as jf:
        idx_to_class = {int(v): k for k, v in json.load(jf).items()}

    return model, idx_to_class


model, idx_to_class = load_model_and_labels()


# ===========================
# 2. Transform definieren (muss identisch zum Training sein!)
# ===========================

transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((48, 48)),
    T.RandomApply([T.AutoAugment()], p=0.2),
    T.ToTensor(),
    T.Normalize(mean=0.5077, std=0.2551)
])


# ===========================
# 3. Funktion: Gesicht extrahieren mit MediaPipe
# ===========================

def extract_first_face(pil_image):

    mp_face_detection = mp.solutions.face_detection

    np_image = np.array(pil_image) # (H,W,C)

    with mp_face_detection.FaceDetection() as fd:
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
# 4. Funktion: Emotion vorhersagen
# ===========================

def predict_emotion(pil_image):

    face_pil = extract_first_face(pil_image)

    if face_pil is None:
        return None, None

    input_tensor = transform(face_pil).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, class_idx = torch.max(probs, dim=1)
        emotion = idx_to_class[class_idx.item()]

        return emotion, confidence.item()


# ===========================
# 5. Streamlit UI
# ===========================

st.title("Emotion Detection")
st.write("Upload a File")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded image", use_container_width=True)

    # Vorhersage
    with st.spinner("Erkenne Gesicht und Emotion..."):
        emotion, confidence = predict_emotion(img)

    if emotion:
        st.success(f"**Vorhergesagte Emotion**: {emotion}")
        st.info(f"**Konfidenz**: {confidence:.2f}")
    else:
        st.error("Kein Gesicht erkannt. Bitte versuche ein anderes Bild.")
