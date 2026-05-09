# Emotion Recognition App

Diese App nutzt ein CNN, um automatisch Emotionen in Gesichtern zu erkennen.

### **Features**
* **Auto-Face-Crop:** Erkennt und isoliert Gesichter via MediaPipe.
* **7 Emotionen:** Wütend, Ekel, Angst, Glücklich, Traurig, Überrascht, Neutral.
* **Tech-Stack:** PyTorch, FastAPI (API), Flet (Web-UI), Docker.

---

## Modell-Architektur
Das Herzstück ist ein **ResNet-basiertes CNN**, das speziell für 48x48 Graustufenbilder optimiert wurde:



* **Residual Blocks:** Nutzen Skip-Connections, um tieferes Lernen ohne Informationsverlust zu ermöglichen.
* **Komponenten:** 3x3 Faltungen, BatchNorm zur Stabilisierung, LeakyReLU Aktivierung und Dropout zur Vermeidung von Overfitting.
* **Layers:** 4 Hauptblöcke mit steigender Kanaltiefe (64, 128, 256, 512).
* **Classifier:** Global Average Pooling gefolgt von einem Fully-Connected-Layer-Head.



---

## Quickstart

**Voraussetzung:** Deine Modell-Datei liegt in `models/best_model.pth`.

1. **Starten:**
   ```bash
   docker-compose up --build
