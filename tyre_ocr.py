import cv2
import numpy as np
import pytesseract
import re


def preprocess_block(img_block, save_debug=False, idx=0):
    gray = cv2.cvtColor(img_block, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel_sharpen)

    denoised = cv2.fastNlMeansDenoising(sharp, None, 15, 7, 21)

    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

    if save_debug:
        cv2.imwrite(f'block_{idx}_preprocessed.png', morph)

    return morph


def split_image_into_blocks(image_path, block_width=500, overlap=150, save_debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")

    h, w = img.shape[:2]
    blocks = []
    start = 0

    while start < w:
        end = min(start + block_width, w)
        block = img[:, start:end]
        if save_debug:
            cv2.imwrite(f'block_{start}_{end}.png', block)
        blocks.append(block)
        start += (block_width - overlap)

    return blocks


def clean_ocr_text(text):
    text = text.upper()
    replacements = {'O': '0', 'I': '1', 'L': '1', '|': '1', 'S': '5'}
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r'[^A-Z0-9/.\- ]', '', text)
    return text


def postprocess_text(text):
    text = clean_ocr_text(text)
    patterns = {
        'model': r'(WANDERER(?:\s*ECOTRED|\s*STREET)?|ECOTRED|STREET)',
        'size': r'(\d{3}/\d{2}\s?R\d{2}\s?\d{2}[A-Z]?(?:\s?TL)?)',
        'brand': r'(MRF)',
        'steel': r'(STEEL\s*BELTED\s*RADIAL)',
        'tubeless': r'(TUBELESS|TL)',
        'dot': r'(DOT\s*\w{2}\s*\w{2}\s*\w{2,4}\s*\d{4})',
    }
    results = {}
    for key, pat in patterns.items():
        matches = re.findall(pat, text, re.IGNORECASE)
        results[key] = ', '.join(match.upper() for match in matches) if matches else ''
    output = []
    for key in ['model', 'size', 'brand', 'steel', 'tubeless', 'dot']:
        if results[key]:
            if key == 'dot':
                output.append(results[key])
            else:
                output.append(results[key].title())
    return '\n'.join(output)


def extract_text_from_blocks(image_path):
    blocks = split_image_into_blocks(image_path, block_width=500, overlap=150, save_debug=True)
    all_texts = []

    for idx, block in enumerate(blocks):
        processed = preprocess_block(block, save_debug=True, idx=idx)

        config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/.-R'
        text_normal = pytesseract.image_to_string(processed, config=config)
        text_inverted = pytesseract.image_to_string(cv2.bitwise_not(processed), config=config)

        chosen_text = text_normal if len(text_normal.strip()) >= len(text_inverted.strip()) else text_inverted
        if chosen_text.strip():
            all_texts.append(chosen_text.strip())

    return ' '.join(all_texts)


def main():
    image_path = r"F:\Projects-but edhum panna maten\Intern 2025 fall\MRF.png"
    combined_text = extract_text_from_blocks(image_path)
    print("----- OCR RAW TEXT -----")
    print(combined_text)
    print("----- POSTPROCESSED TYRE INFO -----")
    print(postprocess_text(combined_text))


if __name__ == "__main__":
    main()
