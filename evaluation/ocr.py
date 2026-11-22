from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageOps
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
    """Split a page into approximate text-line crops using horizontal ink projection."""
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    # Ignore faint paper texture while retaining ordinary dark pen/pencil strokes.
    ink = gray < 185
    row_ink = ink.sum(axis=1)
    active = row_ink >= max(2, int(width * 0.002))
    rows = np.flatnonzero(active)
    if rows.size == 0:
        return [image]

    # Join nearby ink rows so dots, accents and letter gaps stay with their line.
    groups = []
    start = previous = int(rows[0])
    max_gap = max(12, int(height * 0.008))
    for row in rows[1:]:
        row = int(row)
        if row - previous > max_gap:
            groups.append((start, previous))
            start = row
        previous = row
    groups.append((start, previous))

    padding = max(8, int(height * 0.006))
    crops = []
    for top, bottom in groups:
        if bottom - top < 3:
            continue
        y1 = max(0, top - padding)
        y2 = min(height, bottom + padding + 1)
        crops.append(image.crop((0, y1, width, y2)))
    return crops or [image]


def recognise_handwriting(image_file):
    """Recognise handwritten text from an uploaded page or cropped text region.

    The TrOCR checkpoint is line-oriented. This function first estimates
    horizontal text lines from the page image, then recognises each line in
    reading order. Teachers should still review the result for OCR mistakes.
    """
    image = Image.open(image_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Avoid feeding enormous camera images into line segmentation.
    image.thumbnail((2200, 3000))
    line_images = segment_handwritten_lines(image)
    processor, model, device = load_handwriting_model()

    recognised_lines = []
    for line_image in line_images:
        pixel_values = processor(images=line_image, return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            generated_ids = model.generate(pixel_values, max_new_tokens=128)
        line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if line_text:
            recognised_lines.append(line_text)
    return "\n".join(recognised_lines)
