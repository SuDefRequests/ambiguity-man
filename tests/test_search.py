"""HTTP contract tests with real exports and a mocked Chroma query boundary.

These validate hydration/filter forwarding, not live embedding retrieval quality.
"""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from app.api import app


class SearchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.records = []
        for name in ('baws', 'cad'):
            with (root / f'{name}_data_for_graph.json').open(encoding='utf-8') as source:
                cls.records.extend(json.load(source))
        cls.by_id = {r['chunk_id']: r for r in cls.records}

    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.collection = Mock()
        self.patch = patch('app.api.get_collection', return_value=self.collection)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.groq = patch('app.api.get_groq_client', side_effect=AssertionError('No LLM calls'))
        self.groq.start()
        self.addCleanup(self.groq.stop)

    def seed_hits(self, archive='all', volume=None, count=20):
        records = [r for r in self.records if 'education' in r['text'].lower()
                   and (archive == 'all' or r['archive_type'].lower() == archive)
                   and (volume is None or r['volume'] == volume)][:count]
        self.assertTrue(records)
        self.collection.query.return_value = {'ids': [[r['chunk_id'] for r in records]],
                                              'documents': [['untrusted index text']]}
        return records

    def test_basic_search_hydrates_exact_exports_and_reader_links(self):
        self.seed_hits()
        response = self.client.get('/api/v1/search', params={'q': 'education'})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {'query', 'archive_searched', 'total_results', 'results'})
        self.assertEqual(body['total_results'], 8)
        for hit in body['results']:
            record = self.by_id[hit['passage_id']]
            self.assertIn('education', hit['text'].lower())
            self.assertEqual(hit['snippet'], record['text'][:300])
            self.assertIsNone(hit['relevance_score'])
            reader = self.client.get('/api/v1/passages/' + hit['passage_id'])
            self.assertEqual(reader.status_code, 200)
            self.assertEqual({k: v for k, v in hit.items() if k not in ('snippet', 'relevance_score')}, reader.json())
        self.collection.query.assert_called_once_with(query_texts=['education'], n_results=8,
            where={'archive_type': {'$in': ['BAWS', 'CAD', 'ocr']}})

    def test_filters_are_sent_before_retrieval_and_metadata_is_preserved(self):
        for archive in ('all', 'baws', 'cad'):
            for volume in (None, 1, 3, 5):
                with self.subTest(archive=archive, volume=volume):
                    records = self.seed_hits(archive, volume, 3)
                    params = dict(q='education', archive=archive, limit=3)
                    if volume is not None:
                        params['volume'] = volume
                    response = self.client.get('/api/v1/search', params=params)
                    self.assertEqual(response.status_code, 200)
                    hits = response.json()['results']
                    self.assertEqual([h['passage_id'] for h in hits], [r['chunk_id'] for r in records])
                    for hit, record in zip(hits, records):
                        for key in ('text', 'source', 'page', 'volume', 'title', 'url'):
                            self.assertEqual(hit[key], record[key])
                        self.assertEqual(hit['archive_type'], record['archive_type'].lower())
                    expected = {'archive_type': {'$in': ['BAWS', 'CAD', 'ocr']}} if archive == 'all' else {'archive_type': archive.upper()}
                    if volume is not None:
                        expected = {'$and': [expected, {'volume': volume}]}
                    self.assertEqual(self.collection.query.call_args.kwargs['where'], expected)

    def test_limit(self):
        for limit in (1, 8, 20):
            self.seed_hits(count=20)
            body = self.client.get('/api/v1/search', params={'q': 'education', 'limit': limit}).json()
            self.assertEqual(len(body['results']), limit)
            self.assertEqual(body['total_results'], limit)
            self.assertEqual(self.collection.query.call_args.kwargs['n_results'], limit)

    def test_empty_retrieval(self):
        self.collection.query.return_value = {'ids': [[]]}
        response = self.client.get('/api/v1/search', params={'q': 'no-match'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), dict(query='no-match', archive_searched='all', total_results=0, results=[]))

    def test_missing_blank_and_invalid_parameters(self):
        cases = [({}, 'q'), ({'q': ''}, 'q'), ({'q': '  \t'}, 'q')]
        for key, values in {'archive': ['bad'], 'volume': [0, 6, 'x', '1.5'],
                            'limit': [0, 21, 'x', '1.5']}.items():
            cases.extend(({'q': 'education', key: value}, key) for value in values)
        for params, field in cases:
            with self.subTest(params=params):
                response = self.client.get('/api/v1/search', params=params)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertTrue(any(e['loc'] == ['query', field] for e in response.json()['detail']))
        self.collection.query.assert_not_called()

    def test_query_trimmed(self):
        self.seed_hits()
        body = self.client.get('/api/v1/search', params={'q': '  education  '}).json()
        self.assertEqual(body['query'], 'education')

    def test_stale_duplicate_and_wrong_filter_hits_are_not_exposed(self):
        good = self.seed_hits('cad', 2, 2)
        wrong = next(r for r in self.records if r['archive_type'] == 'BAWS')
        self.collection.query.return_value = {'ids': [['upload-only', wrong['chunk_id'], good[0]['chunk_id'], good[0]['chunk_id'], good[1]['chunk_id']]]}
        response = self.client.get('/api/v1/search', params=dict(q='education', archive='cad', volume=2))
        self.assertEqual(response.status_code, 200)
        self.assertEqual([h['passage_id'] for h in response.json()['results']], [r['chunk_id'] for r in good])

    def test_backend_failure_is_not_an_empty_result(self):
        self.collection.query.side_effect = RuntimeError('private backend details')
        response = self.client.get('/api/v1/search', params={'q': 'education'})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()['detail']['code'], 'search_unavailable')
        self.assertNotIn('private', response.text)
