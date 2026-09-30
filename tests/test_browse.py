"""Browse contract checks against the actual exports, without search services."""
import json
from pathlib import Path
import unittest

from fastapi.testclient import TestClient
from app.api import app

ROOT = Path(__file__).resolve().parents[1]


class BrowseContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.exports = {}
        for archive in ('baws', 'cad'):
            with (ROOT / f'{archive}_data_for_graph.json').open(encoding='utf-8') as source:
                cls.exports[archive] = json.load(source)
        cls.exports['all'] = cls.exports['baws'] + cls.exports['cad']

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def assert_page(self, archive='all', offset=0, limit=8, volume=None):
        params = dict(archive=archive, offset=offset, limit=limit)
        records = self.exports[archive]
        if volume is not None:
            params['volume'] = volume
            records = [r for r in records if r['volume'] == volume]
        response = self.client.get('/api/v1/passages', params=params)
        self.assertEqual(response.status_code, 200, response.text)
        expected = [dict(
            passage_id=r['chunk_id'], archive_type=r['archive_type'].lower(),
            **{key: r[key] for key in ('source', 'page', 'volume', 'title', 'url', 'text')},
        ) for r in records[offset:offset + limit]]
        self.assertEqual(response.json(), dict(
            archive=archive, offset=offset, limit=limit, total=len(records),
            has_more=bool(records[offset + limit:]), results=expected,
        ))
        return response.json()

    def test_first_pages_and_truthful_totals(self):
        for archive in ('baws', 'cad', 'all'):
            with self.subTest(archive=archive):
                self.assert_page(archive)

    def test_defaults(self):
        self.assertEqual(self.client.get('/api/v1/passages').json(), self.assert_page())

    def test_second_pages_are_next_source_records_without_duplicates(self):
        for archive in ('baws', 'cad', 'all'):
            with self.subTest(archive=archive):
                first = self.assert_page(archive)['results']
                second = self.assert_page(archive, offset=8)['results']
                self.assertTrue({r['passage_id'] for r in first}.isdisjoint(
                    r['passage_id'] for r in second))

    def test_all_preserves_baws_then_cad_boundary(self):
        self.assert_page('all', offset=len(self.exports['baws']) - 3)

    def test_each_volume_filter_preserves_order_and_total(self):
        for archive in ('baws', 'cad', 'all'):
            for volume in range(1, 6):
                with self.subTest(archive=archive, volume=volume):
                    self.assert_page(archive, volume=volume)
                    self.assert_page(archive, volume=volume, offset=8)

    def test_final_and_empty_pages(self):
        for archive in ('baws', 'cad', 'all'):
            for volume in (None, 3):
                records = self.exports[archive]
                total = sum(volume is None or r['volume'] == volume for r in records)
                with self.subTest(archive=archive, volume=volume):
                    final = self.assert_page(archive, offset=total - 3, volume=volume)
                    self.assertEqual(len(final['results']), 3)
                    self.assertFalse(final['has_more'])
                    self.assert_page(archive, offset=total - 8, volume=volume)
                    self.assert_page(archive, offset=total, volume=volume)
                    self.assert_page(archive, offset=total + 100, volume=volume)

    def test_allowed_limit_boundaries(self):
        self.assert_page(limit=1)
        self.assert_page(limit=20)

    def test_invalid_parameters_return_structured_422(self):
        cases = [('archive', 'unknown'), ('archive', 'BAWS'), ('volume', 0),
                 ('volume', 6), ('volume', 'text'), ('volume', '1.5'),
                 ('offset', -1), ('offset', 'text'), ('offset', '1.5'),
                 ('limit', 0), ('limit', 21), ('limit', 'text'), ('limit', '1.5')]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                response = self.client.get('/api/v1/passages', params={field: value})
                self.assertEqual(response.status_code, 422)
                self.assertTrue(any(e['loc'] == ['query', field] for e in response.json()['detail']))

    def test_results_equal_passage_by_id_contract(self):
        for archive in ('baws', 'cad'):
            for passage in self.assert_page(archive)['results']:
                response = self.client.get('/api/v1/passages/' + passage['passage_id'])
                self.assertEqual(response.status_code, 200)
                self.assertEqual(passage, response.json())
        schema = self.client.get('/openapi.json').json()['components']['schemas']
        self.assertEqual(schema['ArchiveBrowseResponse']['properties']['results']['items'],
                         {'$ref': '#/components/schemas/ArchivePassage'})

    def test_repeated_request_is_deterministic(self):
        params = dict(archive='cad', volume=2, offset=8, limit=20)
        first = self.client.get('/api/v1/passages', params=params)
        second = self.client.get('/api/v1/passages', params=params)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json(), second.json())


if __name__ == '__main__':
    unittest.main()
