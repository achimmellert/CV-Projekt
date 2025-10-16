# export_onnx.py
import torch
from SimpleCNN import CNN  # Ersetze 'your_module' mit dem tatsächlichen Modulnamen (z.B. 'main')

# Lade dein trainiertes Modell
model = CNN(num_classes=7, dropout_b=0.2, dropout_fc=0.4)
model.load_state_dict(torch.load("deploy1/best_simple_cnn_model.pth", map_location="cpu"))
model.eval()

# Dummy-Eingabe (48×48 Graustufen)
dummy_input = torch.randn(1, 1, 48, 48)

# Exportiere als ONNX
torch.onnx.export(
    model,
    dummy_input,
    "emotion_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✅ ONNX-Modell erfolgreich exportiert als 'emotion_model.onnx'")