import os
import sys
import uuid
from typing import List, Tuple

from PIL import Image

# Reuse the OCR pipeline from main.py
from main import initialize_model, run_ocr, latex_to_readable, select_device


def normalize_tokens(s: str) -> str:
    """Light normalization to fix common OCR spacing artifacts."""
    import re

    # Fix split/garbled 'TREADWEAR' variants (e.g., 'TREAD WEA R', 'TREAD ... EAR', 'TREADWE')
    s = re.sub(r"(?i)\bTREAD\s*W?E?A?R\b", "TREADWEAR", s)
    s = re.sub(r"(?i)\bTREAD\s*[^\d\w]{0,6}\s*WE?A?R\b", "TREADWEAR", s)
    # Also handle 'TREAD' ... 'EAR' separated by noise
    s = re.sub(r"(?i)\bTREAD\b[\s\w]{0,10}\bEAR\b", "TREADWEAR", s)

    # Ensure proper UTQG spacing
    s = re.sub(r"(?i)\bTREADWEAR\s*(\d+)", r"TREADWEAR \1", s)
    s = re.sub(r"(?i)\bTRACTION\s*([A-C][+\-]?)\b", r"TRACTION \1", s)
    s = re.sub(r"(?i)\bTEMPERATURE\s*([A-C])\b", r"TEMPERATURE \1", s)

    # Normalize tire size patterns like 205/60 R16 92 H -> 205/60R16 92H
    s = re.sub(r"\b(\d{3})\s*/\s*(\d{2})\s*R\s*(\d{2})\b", r"\1/\2R\3", s)
    s = re.sub(r"\b(\d{3})\s*/\s*(\d{2})\s*R(\d{2})\s*(\d{2})\s*([A-Z])\b", r"\1/\2R\3 \4\5", s)

    # Common phrases
    s = re.sub(r"(?i)\bSTEEL\s*BELTED\s*RADIAL\b", "STEEL BELTED RADIAL", s)
    s = re.sub(r"(?i)\bTUBE\s*LESS\b", "TUBELESS", s)
    s = re.sub(r"(?i)\bMADE\s*IN\s*INDIA\b", "MADE IN INDIA", s)

    # Units normalization: map mis-OCRed kPa/psi
    s = re.sub(r"(?i)\b[kv][pP][aAsS]\b", "KPA", s)  # kpa, Vp a, etc.
    s = re.sub(r"(?i)\b(p\s*s|p\s*\.?\s*s\.?|psi)\b", "PSI", s)
    s = re.sub(r"(?i)\bkg\s*s?\b", "KG", s)

    # Common OCR fixes
    s = re.sub(r"(?i)\bMAX\.?\s*LOA\b", "MAX LOAD", s)
    s = re.sub(r"(?i)\bSIDE\s*WALL\b", "SIDEWALL", s)
    s = re.sub(r"(?i)\bSAF[EI]TY\s*WARNING\b", "SAFETY WARNING", s)
    s = re.sub(r"(?i)\bPOLY\s*ESTER\b", "POLYESTER", s)
    s = re.sub(r"(?i)\bBELT\s*ED\b", "BELTED", s)

    # Collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_fields(text: str) -> dict:
    """Extract key tire fields using regex heuristics."""
    import re
    u = text.upper()
    fields = {
        "size": None,            # e.g., 205/60R16 92H
        "treadwear": None,       # e.g., 360
        "traction": None,        # e.g., A
        "temperature": None,     # e.g., B
        "dot": None,             # DOT code snippet (short)
        "dot_full": None,        # DOT code full
        "construction": None,    # STEEL BELTED RADIAL
        "tubeless": None,        # TUBELESS
        "country": None,         # MADE IN INDIA
        "max_load": None,        # e.g., MAX LOAD 630 kg
        "max_pressure": None,    # e.g., MAX PRESSURE 350 KPA
        "max_pressure_psi": None,# e.g., (51 PSI)
        "brand": None,           # e.g., MRF
        "model": None,           # e.g., ECOTRED
        "tread_ply": None,       # e.g., TREAD: ... PLY ...
        "sidewall_ply": None,    # e.g., SIDEWALL: ... PLY ...
        "warning": None,         # SAFETY WARNING: ...
    }

    m = re.search(r"(\d{3})\s*/\s*(\d{2})\s*R\s*(\d{2})(?:\s*(\d{2})\s*([A-Z]))?", u)
    if m:
        size = f"{m.group(1)}/{m.group(2)}R{m.group(3)}"
        if m.group(4) and m.group(5):
            size += f" {m.group(4)}{m.group(5)}"
        fields["size"] = size

    m = re.search(r"TREAD\s*WEAR\s*(\d{2,4})|TREADWEAR\s*(\d{2,4})", u)
    if m:
        fields["treadwear"] = m.group(1) or m.group(2)

    m = re.search(r"TRACTION\s*([A-C][+\-]?)", u)
    if m:
        fields["traction"] = m.group(1)

    m = re.search(r"TEMPERATURE\s*([A-C])", u)
    if m:
        fields["temperature"] = m.group(1)

    m = re.search(r"\bDOT\b\s*([A-Z0-9\s-]{5,})", u)
    if m:
        dot_tail = m.group(1).strip()
        fields["dot_full"] = ("DOT " + dot_tail)
        short = ("DOT " + dot_tail).split()[0:5]
        fields["dot"] = " ".join(short)

    if re.search(r"STEEL\s*BELTED\s*RADIAL", u):
        fields["construction"] = "STEEL BELTED RADIAL"
    if re.search(r"TUBE\s*LESS", u):
        fields["tubeless"] = "TUBELESS"
    if re.search(r"MADE\s*IN\s*INDIA", u):
        fields["country"] = "MADE IN INDIA"

    m = re.search(r"MAX\.?\s*LOAD\s*([0-9]{2,5})\s*(KG|KGS|LB|LBS)", u)
    if m:
        fields["max_load"] = f"MAX LOAD {m.group(1)} {m.group(2)}".replace("KGS", "KG")

    # Catch noisy kPa/psi variants (KPA, VPA, VPS -> KPA)
    m = re.search(r"MAX\.?\s*PRESSURE\s*([0-9]{2,4})\s*((?:K|V)?PA|PSI)", u)
    if m:
        unit = m.group(2)
        unit = "KPA" if unit.upper().endswith("PA") else "PSI"
        fields["max_pressure"] = f"MAX PRESSURE {m.group(1)} {unit}"

    # Parenthetical PSI
    m = re.search(r"\((\d{2,3})\s*PSI\)", u)
    if m:
        fields["max_pressure_psi"] = f"{m.group(1)} PSI"

    # Brand and model
    m = re.search(r"\b(MRF|MICHELIN|BRIDGESTONE|GOODYEAR|CEAT|APOLLO|CONTINENTAL|PIRELLI|YOKOHAMA|HANKOOK|DUNLOP)\b", u)
    if m:
        fields["brand"] = m.group(1).title()
    m = re.search(r"\b(ECO\s*TRED|ECOTRED|ECOTREAD|ECOTR\w+)\b", u)
    if m:
        fields["model"] = re.sub(r"\s+", "", m.group(1)).upper()

    # Ply and warning lines (on raw text to keep words)
    tread_line = re.search(r"(?i)\bTREAD\b[^\n]{0,120}", text)
    if tread_line:
        fields["tread_ply"] = normalize_tokens(tread_line.group(0))
    side_line = re.search(r"(?i)\bSIDE\s*WALL\b[^\n]{0,120}|\bSIDEWALL\b[^\n]{0,120}", text)
    if side_line:
        fields["sidewall_ply"] = normalize_tokens(side_line.group(0))
    warn_line = re.search(r"(?i)SAF[EI]TY\s*WARNING[^\n]{0,160}", text)
    if warn_line:
        fields["warning"] = normalize_tokens(warn_line.group(0))

    return fields


def synthesize_summary(strips: list) -> str:
    """Aggregate fields from all strips and build a canonical summary string."""
    aggregate = {}
    for s in strips:
        f = extract_fields(s)
        for k, v in f.items():
            if v and not aggregate.get(k):
                aggregate[k] = v

    lines = []
    # Header: BRAND MODEL SIZE
    header = []
    if aggregate.get("brand"):
        header.append(aggregate["brand"])
    if aggregate.get("model"):
        header.append(aggregate["model"])
    if aggregate.get("size"):
        header.append(aggregate["size"])
    if header:
        lines.append(" ".join(header))
    elif aggregate.get("size"):
        lines.append(aggregate["size"])
    utqg = []
    if aggregate.get("treadwear"):
        utqg.append(f"TREADWEAR {aggregate['treadwear']}")
    if aggregate.get("traction"):
        utqg.append(f"TRACTION {aggregate['traction']}")
    if aggregate.get("temperature"):
        utqg.append(f"TEMPERATURE {aggregate['temperature']}")
    if utqg:
        lines.append(" ".join(utqg))
    if aggregate.get("construction"):
        lines.append(aggregate["construction"])
    if aggregate.get("tubeless"):
        lines.append(aggregate["tubeless"])
    if aggregate.get("max_load"):
        lines.append(aggregate["max_load"])
    if aggregate.get("max_pressure"):
        if aggregate.get("max_pressure_psi"):
            lines.append(f"{aggregate['max_pressure']} ({aggregate['max_pressure_psi']})")
        else:
            lines.append(aggregate["max_pressure"])
    if aggregate.get("country"):
        lines.append(aggregate["country"])
    if aggregate.get("tread_ply"):
        lines.append(aggregate["tread_ply"])
    if aggregate.get("sidewall_ply"):
        lines.append(aggregate["sidewall_ply"])
    if aggregate.get("warning"):
        lines.append(aggregate["warning"])
    if aggregate.get("dot_full"):
        lines.append(aggregate["dot_full"])
    elif aggregate.get("dot"):
        lines.append(aggregate["dot"])

    # Deduplicate lines (case-insensitive), preserve order
    seen = set()
    unique_lines = []
    for ln in lines:
        key = ln.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique_lines.append(ln)

    return "\n".join(unique_lines)


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def expand_box(box: Tuple[int, int, int, int], padding: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        clamp(x1 - padding, 0, w),
        clamp(y1 - padding, 0, h),
        clamp(x2 + padding, 0, w),
        clamp(y2 + padding, 0, h),
    )


def split_into_quadrants(img: Image.Image, padding: int = 24) -> List[Image.Image]:
    """
    Split the image into 4 vertical strips and expand each crop by `padding` pixels
    on all sides (clamped to image boundaries) to avoid cutting characters.
    Order: strip-1 (leftmost) -> strip-4 (rightmost).
    """
    w, h = img.size
    step = w // 4
    x0 = 0
    x1 = step
    x2 = step * 2
    x3 = step * 3
    x4 = w

    boxes = [
        (x0, 0, x1, h),  # strip 1
        (x1, 0, x2, h),  # strip 2
        (x2, 0, x3, h),  # strip 3
        (x3, 0, x4, h),  # strip 4
    ]

    expanded = [expand_box(b, padding, w, h) for b in boxes]
    return [img.crop(b) for b in expanded]


def save_temp_images(crops: List[Image.Image]) -> List[str]:
    temp_dir = os.path.join(os.getcwd(), "temp_crops")
    os.makedirs(temp_dir, exist_ok=True)
    uid = uuid.uuid4().hex
    paths: List[str] = []
    for idx, crop in enumerate(crops, start=1):
        path = os.path.join(temp_dir, f"{uid}_q{idx}.png")
        crop.save(path)
        paths.append(path)
    return paths


def prompt_inputs() -> Tuple[str, str, int]:
    print("Enter image path (PNG/JPG):", end=" ")
    image_path = input().strip().strip('"')
    if not image_path:
        print("No image path provided.")
        sys.exit(1)

    print("Select task: [1] Plain Text OCR  [2] Format Text OCR (default 1):", end=" ")
    choice = input().strip()
    task = "Plain Text OCR" if choice != "2" else "Format Text OCR"

    print("Padding pixels around each crop (default 24):", end=" ")
    pad_raw = input().strip()
    try:
        padding = int(pad_raw) if pad_raw else 24
    except ValueError:
        padding = 24

    return image_path, task, padding


def main() -> None:
    image_path, task, padding = prompt_inputs()

    try:
        base_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image: {e}")
        sys.exit(1)

    print(f"\nSplitting into 4 parts with padding={padding} ...")
    crops = split_into_quadrants(base_img, padding=padding)
    crop_paths = save_temp_images(crops)

    print(f"Loading model on {select_device()}...")
    processor, model, device = initialize_model()

    results: List[str] = []
    labels = ["Strip-1", "Strip-2", "Strip-3", "Strip-4"]

    print("\nRunning OCR on each part...\n")
    for label, path in zip(labels, crop_paths):
        raw = run_ocr(image_path=path, task=task, processor=processor, model=model, device=device)
        clean = normalize_tokens(latex_to_readable(raw))
        results.append(f"[{label}]\n{clean}")

    print("Result (concatenated):\n")
    print("\n\n".join(results))

    # Canonical synthesized summary from all strips
    print("\nCanonical summary:\n")
    clean_strips = [normalize_tokens(latex_to_readable(run_ocr(image_path=p, task=task, processor=processor, model=model, device=device))) for p in crop_paths]
    print(synthesize_summary(clean_strips))

    # Optional: keep temp crops for debugging; uncomment to delete
    # import shutil
    # shutil.rmtree(os.path.dirname(crop_paths[0]), ignore_errors=True)


if __name__ == "__main__":
    main()


