import hashlib
import io
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from PIL import Image
import pymupdf
from fastapi.testclient import TestClient
from app.api import app
from app.ocr.storage import OCRStorage
from app.ocr.models import IngestRequest
from app.ocr.ingestion import build_passages, chunk_text

class OCRTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict('os.environ', {'OCR_DATA_DIR': self.tmp.name})
        self.env.start(); self.addCleanup(self.env.stop)
        self.engine = patch('app.ocr.service.recognize', return_value=('Archival education text', .93))
        self.engine.start(); self.addCleanup(self.engine.stop)
        self.version = patch('app.ocr.service.version', return_value='3.3.2')
        self.version.start(); self.addCleanup(self.version.stop)
        self.collection = Mock()
        self.index = patch('app.api.get_collection', return_value=self.collection)
        self.index.start(); self.addCleanup(self.index.stop)
        self.client = TestClient(app); self.addCleanup(self.client.close)
        out = io.BytesIO(); Image.new('RGB', (100, 100), 'white').save(out, 'PNG')
        self.image = out.getvalue()

    def preview(self, content=None, filename='scan.png', language='en'):
        return self.client.post('/api/v1/ocr/preview', files={'file': (filename, self.image if content is None else content)}, data={'language': language})

    def payload(self, staged):
        return {'document_id': staged['document_id'], 'metadata': {'title': 'Archive', 'language': 'eng', 'collection': 'manuscripts'},
                'pages': [{'page': p['page'], 'text': p['text']} for p in staged['pages']]}

    def test_image_page_confidence_provenance_and_original(self):
        response = self.preview(language='mr'); self.assertEqual(response.status_code, 200)
        body = response.json(); self.assertEqual(body['pages'], [{'page': 1, 'text': 'Archival education text', 'confidence': .93}])
        self.assertEqual(body['sha256'], hashlib.sha256(self.image).hexdigest())
        self.assertEqual(body['ocr_engine'], 'PaddleOCR'); self.assertEqual(body['ocr_engine_version'], '3.3.2')
        self.assertTrue(body['processing_timestamp']); self.assertEqual(body['language'], 'mr')
        self.assertEqual(OCRStorage().path('originals', body['document_id'], '.png').read_bytes(), self.image)
        self.assertNotIn(self.tmp.name, response.text)

    def test_scanned_pdf_and_multiple_pages(self):
        for count in (1, 3):
            with pymupdf.open() as doc:
                for _ in range(count):
                    doc.new_page().insert_image(pymupdf.Rect(0, 0, 100, 100), stream=self.image)
                response = self.preview(doc.tobytes(), 'scan.pdf')
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual([p['page'] for p in response.json()['pages']], list(range(1, count + 1)))

    def test_tiff_and_jpeg(self):
        for suffix, fmt in [('tif', 'TIFF'), ('jpeg', 'JPEG')]:
            buf = io.BytesIO(); Image.new('RGB', (40, 40)).save(buf, fmt)
            self.assertEqual(self.preview(buf.getvalue(), 'scan.' + suffix).status_code, 200)

    def test_invalid_documents(self):
        for content, filename, code in [(b'hello', 'x.txt', 400), (b'', 'x.png', 400), (b'bad', 'x.pdf', 400), (b'bad', 'x.png', 400)]:
            self.assertEqual(self.preview(content, filename).status_code, code)
        self.assertEqual(self.preview(language='xx').status_code, 422)
        with patch('app.ocr.service.MAX_BYTES', 10):
            self.assertEqual(self.preview().status_code, 413)

    def test_metadata_and_document_validation(self):
        staged = self.preview().json(); payload = self.payload(staged)
        for change in [{'volume': -1}, {'language': 'xx'}, {'bogus': 'bad'}]:
            bad = dict(payload, metadata=change)
            self.assertEqual(self.client.post('/api/v1/ocr/ingest', json=bad).status_code, 422)
        self.assertEqual(self.client.post('/api/v1/ocr/ingest', json=dict(payload, document_id='0'*32)).status_code, 404)
        self.assertEqual(self.client.post('/api/v1/ocr/ingest', json=dict(payload, document_id='../bad')).status_code, 422)
        self.assertEqual(self.client.post('/api/v1/ocr/ingest', json=dict(payload, pages=[{'page': 2, 'text': 'bad'}])).status_code, 422)

    def test_deterministic_ingestion_duplicate_and_search(self):
        staged = self.preview().json(); payload = self.payload(staged)
        req = IngestRequest.model_validate(payload)
        self.assertEqual(build_passages(req, staged), build_passages(req, staged))
        response = self.client.post('/api/v1/ocr/ingest', json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()['duplicate'])
        self.assertTrue(self.client.post('/api/v1/ocr/ingest', json=payload).json()['duplicate'])
        self.collection.upsert.assert_called_once()
        passage = next(iter(OCRStorage().passages().values()))
        self.assertIsNone(passage.source); self.assertIsNone(passage.volume)
        self.assertEqual(passage.provenance['sha256'], staged['sha256'])
        self.assertEqual(passage.ocr_confidence, .93)
        self.collection.query.return_value = {'ids': [[passage.passage_id]]}
        for archive in ('ocr', 'all'):
            search = self.client.get('/api/v1/search', params={'q': 'education', 'archive': archive})
            self.assertEqual(search.status_code, 200, search.text)
            self.assertEqual(search.json()['results'][0]['document_id'], staged['document_id'])
        self.assertEqual(self.collection.query.call_args.kwargs['where'], {'archive_type': {'$in': ['BAWS', 'CAD', 'ocr']}})
        payload['pages'][0]['text'] = 'revision'
        self.assertEqual(self.client.post('/api/v1/ocr/ingest', json=payload).status_code, 409)

    def test_all_combines_export_and_ocr_records(self):
        from app.archive import get_passages
        staged = self.preview().json()
        self.client.post('/api/v1/ocr/ingest', json=self.payload(staged))
        ocr = next(iter(OCRStorage().passages().values()))
        exports = get_passages()
        baws = next(p for p in exports.values() if p.archive_type == 'baws')
        cad = next(p for p in exports.values() if p.archive_type == 'cad')
        self.collection.query.return_value = {'ids': [[baws.passage_id, ocr.passage_id, cad.passage_id]]}
        body = self.client.get('/api/v1/search', params={'q': 'education', 'archive': 'all'}).json()
        self.assertEqual([p['archive_type'] for p in body['results']], ['baws', 'ocr', 'cad'])

    def test_engine_failure_is_clean(self):
        with patch('app.ocr.service.recognize', side_effect=RuntimeError('internal path')):
            response = self.preview()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('internal path', response.text)
        self.assertFalse(list(Path(self.tmp.name).rglob('*')))

    def test_retry_after_index_failure(self):
        staged = self.preview().json(); payload = self.payload(staged)
        self.collection.upsert.side_effect = RuntimeError('private')
        self.assertEqual(self.client.post('/api/v1/ocr/ingest', json=payload).status_code, 503)
        self.assertFalse(OCRStorage().passages())
        self.collection.upsert.side_effect = None
        self.assertEqual(self.client.post('/api/v1/ocr/ingest', json=payload).status_code, 200)

    def test_ask_uses_existing_pipeline(self):
        staged = self.preview().json(); self.client.post('/api/v1/ocr/ingest', json=self.payload(staged))
        passage = next(iter(OCRStorage().passages().values()))
        self.collection.query.return_value = {'ids': [[passage.passage_id]]}
        groq = Mock(); groq.chat.completions.create.return_value.choices = [Mock(message=Mock(content='Grounded answer'))]
        with patch('app.api.get_groq_client', return_value=groq):
            response = self.client.post('/api/v1/ask', json={'question': 'education?', 'archive': 'ocr'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['sources'][0]['archive_type'], 'ocr')

    def test_chunks_keep_boundaries_and_limit(self):
        chunks = chunk_text('a'*1000 + '\n\n' + 'b'*1000)
        self.assertEqual(chunks, ['a'*1000, 'b'*1000])
        self.assertTrue(all(len(c) <= 1200 for c in chunk_text('x'*5000)))
