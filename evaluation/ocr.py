from functools import lru_cache

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


def recognise_handwriting(image_file):
    """Recognise a single handwritten text line from an uploaded image.

    TrOCR's handwritten checkpoint is line-oriented. For best results, upload
    a tightly cropped, clearly lit line or a page image containing one main
    line. Full-page question segmentation is a later pipeline stage.
    """
    image = Image.open(image_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    processor, model, device = load_handwriting_model()
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.inference_mode():
        generated_ids = model.generate(pixel_values, max_new_tokens=128)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()
