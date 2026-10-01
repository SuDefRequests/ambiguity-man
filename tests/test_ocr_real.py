"""Opt-in real Paddle smoke test; can download models, never writes Chroma."""
import io
import os
import unittest
from PIL import Image, ImageDraw, ImageFont
from app.ocr.service import process_document

@unittest.skipUnless(os.getenv('RUN_REAL_OCR') == '1', 'Opt-in PaddleOCR integration')
class RealOCRTest(unittest.TestCase):
    def test_local_raster_scan(self):
        image = Image.new('RGB', (1200, 240), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((30, 70), 'DIGITAL HERITAGE ARCHIVE', fill='black', font=ImageFont.truetype('DejaVuSans.ttf', 48))
        output = io.BytesIO(); image.save(output, 'PNG')
        result = process_document(output.getvalue(), '.png', 'en')
        self.assertEqual(result['page_count'], 1)
        self.assertIn('HERITAGE', result['pages'][0]['text'].upper())
