import os
import re
import sys
from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image


MODEL_NAME = "stepfun-ai/GOT-OCR-2.0-hf"
STOP_STR = "<|im_end|>"


def select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def initialize_model(model_name: str = MODEL_NAME, device: Optional[str] = None):
    device = device or select_device()
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, low_cpu_mem_usage=True, device_map=device
    )
    model = model.eval().to(device)
    return processor, model, device


def run_ocr(
    image_path: str,
    task: str = "Plain Text OCR",
    processor=None,
    model=None,
    device: Optional[str] = None,
) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = device or select_device()
    image = load_image(image_path)

    if task == "Plain Text OCR":
        inputs = processor(image, return_tensors="pt").to(device)
    elif task == "Format Text OCR":
        inputs = processor(image, return_tensors="pt", format=True).to(device)
    else:
        # Fallback to format-aware for other cases in CLI
        inputs = processor(image, return_tensors="pt", format=True).to(device)

    generate_ids = model.generate(
        **inputs,
        do_sample=False,
        tokenizer=processor.tokenizer,
        stop_strings=STOP_STR,
        max_new_tokens=4096,
    )
    output_text = processor.decode(
        generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return output_text


def strip_stop_string(text: str) -> str:
    text = text.strip()
    if text.endswith(STOP_STR):
        text = text[: -len(STOP_STR)]
    return text.strip()


def latex_to_readable(text: str) -> str:
    """
    Convert LaTeX-like OCR output into more readable plain text.
    Heuristics: remove math delimiters and commands, keep human-meaningful text.
    """
    if not text:
        return text

    text = strip_stop_string(text)

    # Unwrap \text{...} blocks first
    text = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", text)

    # Snapshot for UTQG capture before aggressive cleaning
    snapshot_upper = text.upper()
    treadwear_num = None
    traction_grade = None
    temperature_grade = None

    m = re.search(r"TREAD\s*WEAR\s*([0-9]{2,4})|TREADWEAR\s*([0-9]{2,4})", snapshot_upper)
    if m:
        treadwear_num = m.group(1) or m.group(2)

    m = re.search(r"TRACTION\s*([A-C][+\-]?)", snapshot_upper)
    if m:
        traction_grade = m.group(1)

    m = re.search(r"TEMPERATURE\s*([A-C])|TEMPERATURE([A-C])", snapshot_upper)
    if m:
        temperature_grade = (m.group(1) or m.group(2))

    # Remove display/inline math delimiters but keep contents
    text = text.replace("$$", " ")
    text = text.replace("\\(", " ").replace("\\)", " ")
    text = text.replace("\\[", " ").replace("\\]", " ")

    # Replace common LaTeX escaped punctuation
    text = text.replace("\\%", "%").replace("\\#", "#").replace("\\&", "&")
    text = text.replace("\\_", "_").replace("\\^", "^")

    # Simplify superscripts/subscripts like { }^{12} or _{abc}
    text = re.sub(r"\{\s*\}\^\{([^}]*)\}", r"^\1", text)
    text = re.sub(r"_\{([^}]*)\}", r"_\1", text)

    # Drop remaining backslash commands like \\alpha, \\left, \\right, etc.
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)

    # Remove braces left over
    text = text.replace("{", " ").replace("}", " ")

    # Collapse multiple spaces and newlines; normalize spacing around punctuation
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Remove spaces just inside parentheses and quotes
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r'\s+"', '"', text)

    # Trim artifacts like repeated caret/digits blocks from noisy math
    text = re.sub(r"\^\s*([0-9]+)\s*", r"^\1 ", text)

    # UTQG-specific restoration if the grade/number was captured but got dropped
    # Ensure "TEMPERATURE B" style tokens are present
    if temperature_grade:
        if re.search(r"(?i)\bTEMPERATURE\b", text) and not re.search(r"(?i)\bTEMPERATURE\s*[A-C]\b", text):
            text = re.sub(r"(?i)\bTEMPERATURE\b(?!\s*[A-C])", f"TEMPERATURE {temperature_grade}", text, count=1)

    if traction_grade:
        if re.search(r"(?i)\bTRACTION\b", text) and not re.search(r"(?i)\bTRACTION\s*[A-C][+\-]?\b", text):
            text = re.sub(r"(?i)\bTRACTION\b(?!\s*[A-C])", f"TRACTION {traction_grade}", text, count=1)

    if treadwear_num:
        has_treadwear_word = re.search(r"(?i)\bTREAD\s*WEAR\b|\bTREADWEAR\b", text)
        has_treadwear_number = re.search(r"(?i)\bTREAD\s*WEAR\s*\d+\b|\bTREADWEAR\s*\d+\b", text)
        if has_treadwear_word and not has_treadwear_number:
            text = re.sub(r"(?i)\bTREAD\s*WEAR\b|\bTREADWEAR\b", f"TREADWEAR {treadwear_num}", text, count=1)

    return text.strip()


def prompt_user_inputs() -> Tuple[str, str]:
    print("Enter image path (PNG/JPG):", end=" ")
    image_path = input().strip().strip('"')
    if not image_path:
        print("No image path provided.")
        sys.exit(1)

    print("Select task: [1] Plain Text OCR  [2] Format Text OCR (default 1):", end=" ")
    choice = input().strip()
    task = "Plain Text OCR" if choice != "2" else "Format Text OCR"
    return image_path, task


def main() -> None:
    image_path, task = prompt_user_inputs()
    print(f"\nLoading model on {select_device()}...")
    processor, model, device = initialize_model()

    print("Running OCR...\n")
    raw_text = run_ocr(image_path=image_path, task=task, processor=processor, model=model, device=device)
    readable_text = latex_to_readable(raw_text)

    print("Result:\n")
    print(readable_text)


if __name__ == "__main__":
    main()


