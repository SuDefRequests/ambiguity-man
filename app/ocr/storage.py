"""Local storage boundary; original bytes never share the reviewed text location."""
import json
import os
import re
import tempfile
from pathlib import Path
from app.ocr.models import OCRPassage

class OCRStorage:
    def __init__(self, root=None):
        self.root = Path(root or os.getenv('OCR_DATA_DIR', Path(__file__).resolve().parents[2] / 'ocr_data'))

    def path(self, area, document_id, suffix='.json'):
        if not re.fullmatch(r'[0-9a-f]{32}', document_id):
            raise ValueError('Invalid document ID')
        return self.root / area / (document_id + suffix)

    def write_json(self, area, document_id, record):
        target = self.path(area, document_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent, delete=False) as handle:
            json.dump(record, handle, ensure_ascii=False)
            temporary = handle.name
        os.replace(temporary, target)

    def read_json(self, area, document_id):
        return json.loads(self.path(area, document_id).read_text(encoding='utf-8'))

    def save_original(self, document_id, suffix, content):
        target = self.path('originals', document_id, suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(content)

    def passages(self):
        result = {}
        for path in (self.root / 'approved').glob('*.json'):
            for record in json.loads(path.read_text(encoding='utf-8'))['passages']:
                passage = OCRPassage.model_validate(record)
                result[passage.passage_id] = passage
        return result
