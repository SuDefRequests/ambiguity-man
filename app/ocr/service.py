"""Bounded page rendering and lazy PaddleOCR inference."""
import io
from functools import lru_cache
from importlib.metadata import version
import numpy as np
from PIL import Image, ImageOps
from threading import Lock
from app.ocr.models import OCRPage

LANGUAGES = {'en': 'en', 'eng': 'en', 'hi': 'hi', 'hin': 'hi', 'mr': 'mr', 'mar': 'mr'}
EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif'}
_inference_lock = Lock()
MAX_BYTES = 100 * 1024 * 1024
MAX_PAGES = 500
MAX_PIXELS = 25_000_000

class InvalidDocument(ValueError):
    pass

@lru_cache(maxsize=3)
def get_engine(language):
    from paddleocr import PaddleOCR
    # PP-OCRv3 supports en/hi/mr; PaddleOCR validates the installed language map.
    return PaddleOCR(lang=language, ocr_version='PP-OCRv3', device='cpu',
                     use_doc_orientation_classify=False, use_doc_unwarping=False,
                     use_textline_orientation=False)

def recognize(image, language):
    with _inference_lock:
        prediction = next(iter(get_engine(language).predict(np.asarray(image))))
    texts = list(prediction['rec_texts'])
    scores = [float(s) for s in prediction['rec_scores']]
    return '\n'.join(texts), sum(scores) / len(scores) if scores else None

def render_pages(content, suffix):
    try:
        if suffix == '.pdf':
            import pymupdf
            with pymupdf.open(stream=content, filetype='pdf') as doc:
                if doc.needs_pass or not 1 <= len(doc) <= MAX_PAGES:
                    raise InvalidDocument('Encrypted, empty, or oversized PDF')
                for page in doc:
                    scale = 2
                    if page.rect.width * page.rect.height * scale ** 2 > MAX_PIXELS:
                        raise InvalidDocument('Page dimensions exceed limit')
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False, colorspace=pymupdf.csRGB)
                    yield Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        else:
            with Image.open(io.BytesIO(content)) as image:
                expected = {'.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.tif': 'TIFF', '.tiff': 'TIFF'}[suffix]
                if image.format != expected or not 1 <= getattr(image, 'n_frames', 1) <= MAX_PAGES:
                    raise InvalidDocument('Invalid image format or page count')
                for frame in range(getattr(image, 'n_frames', 1)):
                    image.seek(frame)
                    if image.width * image.height > MAX_PIXELS:
                        raise InvalidDocument('Image dimensions exceed limit')
                    image.load()
                    yield ImageOps.exif_transpose(image).convert('RGB')
    except InvalidDocument:
        raise
    except Exception as exc:
        raise InvalidDocument('Malformed PDF or image') from exc

def process_document(content, suffix, language):
    pages = []
    # Render and validate each page before inference; retain boundaries.
    for number, image in enumerate(render_pages(content, suffix), 1):
        text, confidence = recognize(image, LANGUAGES[language])
        pages.append(OCRPage(page=number, text=text, confidence=confidence).model_dump())
    return {'pages': pages, 'page_count': len(pages), 'ocr_engine': 'PaddleOCR',
            'ocr_engine_version': version('paddleocr')}
