"""Ask HTTP contracts: mocked Chroma/Groq, actual canonical export hydration."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from app.api import app
from app.passage_search import retrieve_passages


class AskContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.records = []
        for name in ('baws', 'cad'):
            with (root / f'{name}_data_for_graph.json').open(encoding='utf-8') as source:
                cls.records.append(json.load(source)[0])

    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.collection = Mock()
        self.collection.query.return_value = {'ids': [[r['chunk_id'] for r in self.records]],
                                              'documents': [['DO NOT USE RAW CHROMA TEXT']]}
        self.groq = Mock()
        self.groq.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='Evidence-based answer.'))])
        for target, value in [('get_collection', self.collection), ('get_groq_client', self.groq)]:
            patcher = patch('app.api.' + target, return_value=value)
            setattr(self, target, patcher.start())
            self.addCleanup(patcher.stop)

    def test_answer_and_canonical_sources_in_retrieval_order(self):
        response = self.client.post('/api/v1/ask', json={'question': '  What do these records discuss?  '})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {'question', 'answer', 'sources'})
        self.assertEqual(body['question'], 'What do these records discuss?')
        self.assertEqual(body['answer'], 'Evidence-based answer.')
        expected = [dict(passage_id=r['chunk_id'], archive_type=r['archive_type'].lower(),
                         **{k: r[k] for k in ('source', 'page', 'volume', 'title', 'url', 'text')})
                    for r in self.records]
        self.assertEqual(body['sources'], expected)
        self.assertEqual(body['sources'][0]['title'], '')
        self.assertEqual(body['sources'][0]['url'], '')
        self.assertTrue(body['sources'][1]['url'])
        for source in body['sources']:
            self.assertEqual(source, self.client.get('/api/v1/passages/' + source['passage_id']).json())
        self.collection.query.assert_called_once_with(query_texts=['What do these records discuss?'],
            n_results=6, where={'archive_type': {'$in': ['BAWS', 'CAD', 'ocr']}})

    def test_filter_and_top_k_forwarding_to_existing_retrieval(self):
        with patch('app.passage_search.retrieve_passages', wraps=retrieve_passages) as retrieve:
            response = self.client.post('/api/v1/ask', json={
                'question': 'Question', 'archive': 'cad', 'volume': 1, 'top_k': 2})
            self.assertEqual(response.status_code, 200)
            retrieve.assert_called_once_with(self.collection, 'Question', 'cad', 1, 2)
            self.collection.query.assert_called_once_with(query_texts=['Question'], n_results=2,
                where={'$and': [{'archive_type': 'CAD'}, {'volume': 1}]})
            self.assertEqual(len(response.json()['sources']), 1)
            self.assertEqual(response.json()['sources'][0]['archive_type'], 'cad')

    def test_invalid_requests_rejected_before_services(self):
        cases = [{}, {'question': ''}, {'question': ' \t\n'}, {'question': None}]
        for key, values in {'archive': ['bad'], 'volume': [0, 6], 'top_k': [0, 9]}.items():
            cases.extend({'question': 'Question', key: value} for value in values)
        for body in cases:
            with self.subTest(body=body):
                response = self.client.post('/api/v1/ask', json=body)
                self.assertEqual(response.status_code, 422)
                self.assertIsInstance(response.json()['detail'], list)
        self.get_collection.assert_not_called()
        self.get_groq_client.assert_not_called()

    def test_no_evidence_skips_groq(self):
        self.collection.query.return_value = {'ids': [[]]}
        response = self.client.post('/api/v1/ask', json={'question': 'Question'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['sources'], [])
        self.assertIn('No relevant archive evidence', response.json()['answer'])
        self.get_groq_client.assert_not_called()

    def test_groq_failure_is_clean_503(self):
        self.groq.chat.completions.create.side_effect = RuntimeError('secret-key-internal')
        response = self.client.post('/api/v1/ask', json={'question': 'Question'})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {'detail': {'code': 'generation_unavailable',
            'message': 'Archive answer generation is unavailable'}})
        self.assertNotIn('secret', response.text)

    def test_groq_initialization_failure_is_clean_503(self):
        self.get_groq_client.side_effect = RuntimeError('secret-key-internal')
        response = self.client.post('/api/v1/ask', json={'question': 'Question'})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('secret', response.text)

    def test_empty_generation_is_503(self):
        self.groq.chat.completions.create.return_value.choices[0].message.content = '  '
        self.assertEqual(self.client.post('/api/v1/ask', json={'question': 'Question'}).status_code, 503)

    def test_prompt_uses_canonical_evidence_and_grounding_instructions(self):
        self.client.post('/api/v1/ask', json={'question': 'Question'})
        arguments = self.groq.chat.completions.create.call_args.kwargs
        self.assertEqual(arguments['model'], 'qwen/qwen3.8-27b')
        system, user = arguments['messages']
        self.assertEqual(system['role'], 'system')
        for instruction in ('ONLY', 'insufficient', 'not instructions', 'Do not invent', 'certainty', 'passage_id'):
            self.assertIn(instruction, system['content'])
        evidence = json.loads(user['content'])
        self.assertEqual(evidence['question'], 'Question')
        for source, record in zip(evidence['archive_evidence'], self.records):
            self.assertEqual(source['text'], record['text'])
            self.assertEqual(source['passage_id'], record['chunk_id'])
        self.assertNotIn('DO NOT USE RAW CHROMA TEXT', user['content'])

    def test_retrieval_failures_preserve_search_503(self):
        for initialization in (True, False):
            with self.subTest(initialization=initialization):
                self.get_collection.side_effect = RuntimeError('secret') if initialization else None
                self.collection.query.side_effect = RuntimeError('secret')
                response = self.client.post('/api/v1/ask', json={'question': 'Question'})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()['detail']['code'], 'search_unavailable')
                self.assertNotIn('secret', response.text)
        self.get_groq_client.assert_not_called()
