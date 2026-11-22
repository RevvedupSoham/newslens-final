from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_NAME = "microsoft/trocr-base-handwritten"


@lru_cache(maxsize=1)
def load_handwriting_model():
    """Load the pretrained handwritten-text model once per server process."""
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def segment_handwritten_lines(image):
    """Split a handwritten page into approximate text-line crops."""
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    ink = gray < 175
    row_ink = ink.sum(axis=1).astype(np.float32)
    smoothed = np.convolve(row_ink, np.ones(5, dtype=np.float32) / 5, mode="same")

    # The previous threshold (about 3 pixels for this sample) classified nearly
    # every row as text. Scale the threshold with image width instead.
    threshold = max(12, int(width * 0.02))
    rows = np.flatnonzero(smoothed >= threshold)
    if rows.size == 0:
        return [image]

    groups = []
    start = previous = int(rows[0])
    for row in rows[1:]:
        row = int(row)
        if row - previous > 3:
            groups.append((start, previous))
            start = row
        previous = row
    groups.append((start, previous))

    # A tall band can contain two touching handwritten lines. Split it at a
    # strong valley in the horizontal projection when the evidence supports it.
    typical_height = float(np.median([bottom - top + 1 for top, bottom in groups]))
    refined = []
    for top, bottom in groups:
        band_height = bottom - top + 1
        if band_height > typical_height * 1.45:
            search_start = top + int(band_height * 0.30)
            search_end = bottom - int(band_height * 0.30)
            if search_end > search_start:
                valley = search_start + int(np.argmin(smoothed[search_start:search_end + 1]))
                left_peak = float(smoothed[top:valley].max()) if valley > top else 0
                right_peak = float(smoothed[valley + 1:bottom + 1].max()) if bottom > valley else 0
                reference_peak = min(left_peak, right_peak)
                if reference_peak and smoothed[valley] < reference_peak * 0.60:
                    refined.extend([(top, valley), (valley + 1, bottom)])
                    continue
        refined.append((top, bottom))

    padding = max(8, int(height * 0.015))
    crops = []
    for top, bottom in refined:
        if bottom - top < 7:
            continue
        y1 = max(0, top - padding)
        y2 = min(height, bottom + padding + 1)
        line = image.crop((0, y1, width, y2))

        line_gray = np.asarray(line.convert("L"))
        columns = np.flatnonzero((line_gray < 210).sum(axis=0) >= 2)
        if columns.size:
            x_padding = max(8, int(width * 0.01))
            x1 = max(0, int(columns[0]) - x_padding)
            x2 = min(width, int(columns[-1]) + x_padding + 1)
            line = line.crop((x1, 0, x2, line.height))

        line = ImageOps.autocontrast(line)
        line = ImageEnhance.Sharpness(line).enhance(1.35)
        if line.width < 1200:
            scale = min(2, 1200 / max(1, line.width))
            line = line.resize(
                (max(1, int(line.width * scale)), max(1, int(line.height * scale))),
                Image.Resampling.LANCZOS,
            )
        crops.append(line)

    return crops or [image]


def recognise_handwriting(image_file):
    """Recognise handwritten text and log crop diagnostics without logging student text."""
    image = Image.open(image_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((2200, 3000))
    image = ImageOps.autocontrast(image)

    line_images = segment_handwritten_lines(image)
    crop_sizes = [crop.size for crop in line_images]
    print(
        f"[NewsLens OCR] image_size={image.size}; "
        f"line_crop_count={len(line_images)}; crop_sizes={crop_sizes}"
    )

    processor, model, device = load_handwriting_model()
    recognised_lines = []
    for index, line_image in enumerate(line_images, start=1):
        pixel_values = processor(images=line_image, return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            generated_ids = model.generate(pixel_values, max_new_tokens=128, num_beams=4)
        line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(
            f"[NewsLens OCR] crop={index}/{len(line_images)}; "
            f"recognised_characters={len(line_text)}"
        )
        if line_text:
            recognised_lines.append(line_text)

    return "\n".join(recognised_lines)
