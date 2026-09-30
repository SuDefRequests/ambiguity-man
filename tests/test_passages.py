"""Run: python -m unittest discover -s tests -v (FastAPI + httpx required)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient

from app.api import app

ROOT = Path(__file__).resolve().parents[1]


class PassageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.records = {}
        for filename in ('baws_data_for_graph.json', 'cad_data_for_graph.json'):
            with (ROOT / filename).open(encoding='utf-8') as source:
                cls.records.update((r['chunk_id'], r) for r in json.load(source))

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def assert_export_matches(self, passage_id):
        record = self.records[passage_id]
        response = self.client.get(f'/api/v1/passages/{passage_id}')
        self.assertEqual(response.status_code, 200)
        expected = {
            'passage_id': record['chunk_id'],
            'archive_type': record['archive_type'].lower(),
            **{key: record[key] for key in ('source', 'page', 'volume', 'title', 'url', 'text')},
        }
        self.assertEqual(response.json(), expected)
        self.assertEqual(response.json(), self.client.get(f'/api/v1/passages/{passage_id}').json())
        return response.json()

    def test_baws_preserves_text_and_missing_metadata(self):
        passage = self.assert_export_matches('vol1_p4_c0')
        self.assertEqual(passage['title'], '')
        self.assertEqual(passage['url'], '')

    def test_cad_preserves_text_session_and_url(self):
        passage = self.assert_export_matches('cad_v1_s1_c0')
        self.assertEqual(passage['title'], '09 Dec 1946')
        self.assertEqual(passage['url'], 'https://www.constitutionofindia.net/debates/09-dec-1946/')

    def test_unknown_passage_returns_structured_404(self):
        response = self.client.get('/api/v1/passages/nonexistent-passage')
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {'detail': {
            'code': 'passage_not_found',
            'message': 'Passage not found',
            'passage_id': 'nonexistent-passage',
        }})

    def test_openapi_declares_passage_contract(self):
        schema = self.client.get('/openapi.json').json()
        response = schema['paths']['/api/v1/passages/{passage_id}']['get']['responses']['200']
        self.assertEqual(response['content']['application/json']['schema'], {
            '$ref': '#/components/schemas/ArchivePassage',
        })
        fields = schema['components']['schemas']['ArchivePassage']['properties']
        self.assertEqual(set(fields), {
            'passage_id', 'archive_type', 'source', 'page', 'volume', 'title', 'url', 'text',
        })
        self.assertEqual(fields['page']['type'], 'integer')
        self.assertEqual(fields['volume']['type'], 'integer')
        self.assertEqual(fields['text']['type'], 'string')

    def test_cold_start_without_optional_services_from_other_directory(self):
        # Fail even on an attempted import: no installed service SDK or API key
        # may be necessary for this endpoint. A fresh process avoids cache masking.
        code = '''
import importlib.abc
import sys
class BlockOptionalServices(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'groq', 'chromadb', 'neo4j', 'qdrant_client', 'sentence_transformers', 'psycopg', 'gtts', 'PyPDF2'}:
            raise AssertionError('Unexpected optional dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptionalServices())
from fastapi.testclient import TestClient
from app.api import app
with TestClient(app) as client:
    response = client.get('/api/v1/passages/cad_v1_s1_c0')
    assert response.status_code == 200, response.text
    assert response.json()['passage_id'] == 'cad_v1_s1_c0'
'''
        environment = {**os.environ, 'PYTHONPATH': str(ROOT), 'GROQ_API_KEY': ''}
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-B', '-c', code], cwd=directory,
                env=environment, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
