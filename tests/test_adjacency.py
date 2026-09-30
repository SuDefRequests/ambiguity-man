"""Adjacency contract against source export sequences and exact record data."""
import json
from itertools import groupby
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.api import app


class AdjacencyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.records = []
        for name in ('baws', 'cad'):
            with (root / f'{name}_data_for_graph.json').open(encoding='utf-8') as source:
                cls.records.extend(json.load(source))
        # Contiguous runs, not global groups: do not skip intervening sessions.
        cls.runs = [list(group) for _, group in groupby(cls.records, key=lambda r: (
            r['archive_type'], r['volume'],
            (r['title'], r['url']) if r['archive_type'] == 'CAD' else None,
        ))]
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    @staticmethod
    def expected(record):
        if record is None:
            return None
        return dict(passage_id=record['chunk_id'], archive_type=record['archive_type'].lower(),
                    **{k: record[k] for k in ('source', 'page', 'volume', 'title', 'url', 'text')})

    def assert_neighbors(self, current, previous, following):
        response = self.client.get('/api/v1/passages/' + current['chunk_id'] + '/adjacency')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {
            'current': self.expected(current), 'previous': self.expected(previous),
            'next': self.expected(following),
        })
        return response.json()

    def test_middle_baws(self):
        run = next(r for r in self.runs if r[0]['archive_type'] == 'BAWS' and len(r) >= 3)
        i = len(run) // 2
        self.assert_neighbors(run[i], run[i - 1], run[i + 1])

    def test_middle_cad(self):
        run = next(r for r in self.runs if r[0]['archive_type'] == 'CAD' and len(r) >= 3)
        i = len(run) // 2
        self.assert_neighbors(run[i], run[i - 1], run[i + 1])

    def test_first_and_last_in_every_volume_and_session(self):
        for run in self.runs:
            with self.subTest(first=run[0]['chunk_id'], last=run[-1]['chunk_id']):
                self.assert_neighbors(run[0], None, run[1] if len(run) > 1 else None)
                self.assert_neighbors(run[-1], run[-2] if len(run) > 1 else None, None)

    def test_collection_boundary(self):
        i = next(i for i in range(1, len(self.records))
                 if self.records[i]['archive_type'] != self.records[i - 1]['archive_type'])
        self.assert_neighbors(self.records[i - 1], self.records[i - 2], None)
        self.assert_neighbors(self.records[i], None, self.records[i + 1])

    def test_records_match_passage_endpoint(self):
        for archive in ('BAWS', 'CAD'):
            run = next(r for r in self.runs if r[0]['archive_type'] == archive and len(r) >= 3)
            body = self.assert_neighbors(run[1], run[0], run[2])
            for passage in body.values():
                fetched = self.client.get('/api/v1/passages/' + passage['passage_id'])
                self.assertEqual(fetched.status_code, 200)
                self.assertEqual(passage, fetched.json())

    def test_unknown_passage(self):
        response = self.client.get('/api/v1/passages/nonexistent-passage/adjacency')
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {'detail': {
            'code': 'passage_not_found', 'message': 'Passage not found',
            'passage_id': 'nonexistent-passage',
        }})

    def test_no_search_or_generation_dependency(self):
        with patch('app.api.get_collection', side_effect=AssertionError('No Chroma')), \
             patch('app.api.get_groq_client', side_effect=AssertionError('No Groq')):
            run = self.runs[0]
            self.assert_neighbors(run[1], run[0], run[2])

    def test_response_schema_reuses_archive_passage(self):
        schema = self.client.get('/openapi.json').json()['components']['schemas']
        fields = schema['ArchiveAdjacencyResponse']['properties']
        ref = {'$ref': '#/components/schemas/ArchivePassage'}
        self.assertEqual(fields['current'], ref)
        for direction in ('previous', 'next'):
            self.assertIn(ref, fields[direction]['anyOf'])
            self.assertIn({'type': 'null'}, fields[direction]['anyOf'])
