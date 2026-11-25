import argparse
import os
import pickle
from pathlib import Path

import gradio as gr
from PIL import Image

from src.audit import load_rgb
from src.config import ROOT
from src.predict import Predictor, deployed_checkpoint


CSS = """
.gradio-container { max-width: 1080px !important; margin: auto; }
#intro { padding: 22px 26px; border-radius: 16px; background: #e8f3ed; color: #17392d; }
#intro h1, #intro p { color: #17392d; }
#intro h1 { margin: 0 0 6px; }
#intro p { margin: 5px 0 0; }
"""


def build_app(predictor):
    def classify(filepath):
        if not filepath:
            raise gr.Error("Upload a photograph first.")
        try:
            image = load_rgb(filepath)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise gr.Error("Could not read this photograph. Try a valid JPEG, PNG, or WebP image.") from error
        scores = predictor.predict_image(image)
        preview = image.copy()
        preview.thumbnail((950, 950))
        return preview, max(scores, key=scores.get), scores

    with gr.Blocks(title="Vegetable Classifier", analytics_enabled=False) as demo:
        gr.Markdown("# Vegetable Classifier\nUpload a photograph of one vegetable from the supported classes.\n\n"
                    + " · ".join(predictor.classes) + f"\n\nModel in use: **{predictor.model_name}**", elem_id="intro")
        with gr.Row():
            with gr.Column():
                upload = gr.File(label="Upload a photograph", file_types=[".jpg", ".jpeg", ".png", ".webp"],
                                 type="filepath", elem_id="photo-upload")
                button = gr.Button("Classify photograph", variant="primary")
                preview = gr.Image(label="Uploaded photograph", interactive=False, height=300, buttons=[],
                                   alt_text="Preview of the uploaded vegetable photograph", elem_id="photo-preview")
            with gr.Column():
                prediction = gr.Textbox(label="Predicted vegetable", interactive=False, elem_id="prediction")
                scores = gr.Label(label="Softmax scores", num_top_classes=len(predictor.classes), elem_id="scores")
                gr.Markdown("Scores show how the model ranks the five classes. They can be high for unsupported objects.")
        button.click(classify, inputs=upload, outputs=[preview, prediction, scores], api_name="classify")
        upload.clear(lambda: (None, "", None), outputs=[preview, prediction, scores], queue=False,
                     api_visibility="private")
    return demo


def main():
    parser = argparse.ArgumentParser(description="Run the local vegetable classification app.")
    parser.add_argument("--checkpoint", type=Path, help="Override the validation-selected model.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".work" / "gradio"))
    try:
        checkpoint, expected = (args.checkpoint, None) if args.checkpoint else deployed_checkpoint()
        predictor = Predictor(checkpoint, args.device, expected_sha256=expected)
    except FileNotFoundError as error:
        parser.exit(1, f"{error}\n")
    except (ValueError, KeyError, OSError, RuntimeError, pickle.UnpicklingError) as error:
        parser.exit(1, f"Could not load the vegetable classifier. Check the checkpoint and device settings. {error}\n")
    print(f"Loaded {predictor.model_name} from {predictor.checkpoint} ({predictor.checkpoint_sha256[:12]}).", flush=True)
    build_app(predictor).launch(server_name="127.0.0.1", server_port=args.port, share=False,
                                max_file_size="20mb", run_history=False, footer_links=["settings"],
                                css=CSS, theme=gr.themes.Soft(primary_hue="emerald"))


if __name__ == "__main__":
    main()
