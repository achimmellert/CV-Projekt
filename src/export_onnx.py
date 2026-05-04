import torch
from src.models.cnn import CNN


def export_model():

    model = CNN(num_classes=7, dropout_b=0.1, dropout_fc=0.5)

    checkpoint_path = "models/best_modern_emotion_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    model.eval()

    # Dummy-Eingabe (Batch_size=1, Channel=1, H=48, W=48)
    dummy_input = torch.randn(1, 1, 48, 48)

    onnx_filename = "models/emotion_model_modern.onnx"

    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_filename,
        export_params=True,  # Speichert die trainierten Parameter im File
        opset_version=13,  # Moderner Standard für breite Kompatibilität
        do_constant_folding=True,  # Optimiert das Modell beim Export
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},  # Erlaubt Inferenz mit verschiedenen Batch-Größen
            'output': {0: 'batch_size'}
        }
    )

    print(f"ONNX-Modell erfolgreich exportiert als '{onnx_filename}'")


if __name__ == "__main__":
    export_model()
