import gradio as gr
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image

model_name = "stepfun-ai/GOT-OCR-2.0-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(
    model_name, low_cpu_mem_usage=True, device_map=device
)
model = model.eval().to(device)

stop_str = "<|im_end|>"


def process_image(image_path: str):
    if image_path is None:
        return "Error: No image provided"

    try:
        pil_image = load_image(image_path)
        inputs = processor(pil_image, return_tensors="pt").to(device)
        generate_ids = model.generate(
            **inputs,
            do_sample=False,
            tokenizer=processor.tokenizer,
            stop_strings=stop_str,
            max_new_tokens=4096,
        )
        res = processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return res
    except Exception as e:
        return f"Error: {str(e)}"


def ocr_demo(image_path: str):
    res = process_image(image_path)
    if isinstance(res, str) and res.startswith("Error:"):
        return res
    res = res.replace("\\title", "\\title ")
    return res


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## GOT-OCR 2.0 — Plain Text OCR")
    gr.Markdown(
        """
        This app runs the `stepfun-ai/GOT-OCR-2.0-hf` model to extract plain text from images.
        GOT-OCR 2.0 is a vision-language transformer that combines a high-capacity image encoder
        with a text decoder, trained end-to-end for robust OCR across diverse documents
        (scans, screenshots, forms, tables, and handwriting). Here we use the plain text mode
        to produce clean, linearized text from a single image input.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                image_input = gr.Image(type="filepath", label="Input Image")
                submit_button = gr.Button("Process", variant="primary")

        with gr.Column(scale=1):
            with gr.Group():
                output_markdown = gr.Textbox(label="Text output")

    submit_button.click(
        ocr_demo,
        inputs=[image_input],
        outputs=[output_markdown],
    )


if __name__ == "__main__":
    demo.launch()
