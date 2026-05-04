import flet as ft
import base64
import httpx
import os


API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

EMOTION_EMOJIS = {
    "angry": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

class EmotionDetector(ft.Column):
    def __init__(self):
        super().__init__(horizontal_alignment=ft.CrossAxisAlignment.CENTER)

        # UI Komponenten
        self.image_display = ft.Image(
            src="placeholder.png",
            width=400,
            height=400,
            fit=ft.BoxFit.CONTAIN,
            visible=False,
            border_radius=ft.border_radius.all(10)
        )

        self.result_text = ft.Text(size=24, weight=ft.FontWeight.BOLD)
        self.confidence_text = ft.Text(size=16)
        self.progress_ring = ft.ProgressRing(visible=False)

        self.result_card = ft.Card(
            content=ft.Container(
                content=ft.Column(
                    controls=[self.result_text, self.confidence_text],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                padding=20,
                width=400,
            ),
            visible=False,
            elevation=5
        )

        self.upload_button = ft.Button(
            "Select image & analyze",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=self.process_image,
            style=ft.ButtonStyle(
                padding=20,
                shape=ft.RoundedRectangleBorder(radius=10),
            )
        )

        self.controls = [
            self.upload_button,
            ft.Container(height=20),
            self.progress_ring,
            self.image_display,
            ft.Container(height=20),
            self.result_card
        ]

    async def process_image(self, e: ft.ControlEvent):

        files = await ft.FilePicker().pick_files(
            allow_multiple=False,
            allowed_extensions=["png", "jpg", "jpeg"],
            with_data=True
        )

        if not files or not files[0].bytes:
            return

        self.result_card.visible = False
        self.image_display.visible = False
        self.progress_ring.visible = True
        self.update()

        try:
            img_data = files[0].bytes

            # Bildanzeige über Base64
            self.image_display.src_base64 = base64.b64encode(img_data).decode()
            self.image_display.visible = True

            # API Request
            api_files = {"file": (files[0].name, img_data, "image/jpeg")}
            async with httpx.AsyncClient() as client:
                response = await client.post(API_URL, files=api_files, timeout=30)

            if response.status_code == 200:
                data = response.json()

                if data.get("success"):
                    emotion_str = str(data['emotion']).lower()
                    emoji = EMOTION_EMOJIS.get(emotion_str, "")

                    self.result_text.value = f"{emoji} Emotion: {data['emotion'].capitalize()}"
                    self.result_text.color = ft.Colors.GREEN_700
                    self.confidence_text.value = f"Confidence: {data['confidence'] * 100:.1f}%"
                else:
                    self.result_text.value = f"⚠️ {data.get('error', 'Unknown Error')}"
                    self.result_text.color = ft.Colors.ORANGE_700
                    self.confidence_text.value = "MediaPipe could not detect a face."

            else:
                self.result_text.value = "❌ API Error"
                self.result_text.color = ft.Colors.RED_700
                self.confidence_text.value = f"Status Code: {response.status_code}"

            self.result_card.visible = True

        except Exception as ex:
            self.result_text.value = "❌ system failure"
            self.result_text.color = ft.Colors.RED_700
            self.confidence_text.value = str(ex)
            self.result_card.visible = True

        finally:
            self.progress_ring.visible = False
            self.update()


def main(page: ft.Page):
    page.title = "Emotion Detection Inference"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO)

    page.add(
        ft.SafeArea(
            content=ft.Container(
                content=ft.Column(
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Icon(ft.Icons.FACE_RETOUCHING_NATURAL, size=50, color=ft.Colors.INDIGO_500),
                        ft.Text(
                            value="Emotion Detection",
                            size=32,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.INDIGO_700
                        ),
                        ft.Text(
                            value="Analyze Emotions",
                            size=16,
                            color=ft.Colors.GREY_700,
                            text_align=ft.TextAlign.CENTER
                        ),
                        ft.Divider(height=30, color=ft.Colors.TRANSPARENT),
                        EmotionDetector()
                    ]
                ),
                padding=30,
                alignment=ft.Alignment.CENTER
            )
        )
    )

if __name__ == "__main__":
    ft.run(main)
