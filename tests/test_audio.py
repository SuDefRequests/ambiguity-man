"""Narration HTTP contracts using a mocked ElevenLabs client; no network calls."""
import os
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from app.api import app
from app.models import MAX_NARRATION_TEXT_LENGTH


class AudioContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        environment = patch.dict(os.environ, {'ELEVENLABS_API_KEY': 'test-key',
                                                  'ELEVENLABS_VOICE_ID': 'configured-test-voice-id'})
        environment.start()
        self.addCleanup(environment.stop)
        client_patch = patch('app.audio.elevenlabs_tts.ElevenLabs')
        self.elevenlabs = client_patch.start()
        self.addCleanup(client_patch.stop)
        self.provider_client = self.elevenlabs.return_value
        self.create = self.provider_client.text_to_speech.convert
        self.audio = b'ID3\x00\xffarchive narration\x80'
        self.create.return_value = iter([self.audio[:4], b'', self.audio[4:]])

    def assert_unavailable(self, response):
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {'detail': {
            'code': 'tts_unavailable',
            'message': 'Archive narration is currently unavailable.',
        }})

    def test_success_preserves_audio_and_defaults(self):
        response = self.client.post('/api/v1/audio/speak', json={'text': 'Archive text.'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['content-type'], 'audio/mpeg')
        self.assertEqual(response.headers['content-disposition'],
                         'attachment; filename="archive-narration.mp3"')
        self.assertEqual(response.content, self.audio)
        self.create.assert_called_once_with(model_id='eleven_multilingual_v2', text='Archive text.',
                                            voice_id='configured-test-voice-id',
                                            output_format='mp3_44100_128', request_options={'max_retries': 0})

    def test_explicit_voice_without_config_and_trim_only(self):
        os.environ.pop('ELEVENLABS_VOICE_ID', None)
        text = 'Archive  quotation:\n“Equality.” नमस्ते'
        response = self.client.post('/api/v1/audio/speak', json={
            'text': ' \t' + text + '\n ', 'voice': 'explicit-test-voice-id', 'language': 'hi'})
        self.assertEqual(response.status_code, 200)
        self.create.assert_called_once_with(model_id='eleven_multilingual_v2', text=text,
                                            voice_id='explicit-test-voice-id',
                                            output_format='mp3_44100_128', request_options={'max_retries': 0})

    def test_provider_interface_and_language_default(self):
        provider = Mock()
        provider.synthesize.return_value = self.audio
        with patch('app.api.get_tts_provider', return_value=provider):
            for language in (None, 'hi'):
                body = {'text': ' Text '}
                if language:
                    body['language'] = language
                response = self.client.post('/api/v1/audio/speak', json=body)
                self.assertEqual(response.content, self.audio)
                provider.synthesize.assert_called_with(text='Text', voice='alloy',
                                                       language=language or 'en')
        self.elevenlabs.assert_not_called()

    def test_invalid_text_rejected_before_provider(self):
        cases = [{}, {'text': ''}, {'text': ' \t\n'}, {'text': None},
                 {'text': 'x' * (MAX_NARRATION_TEXT_LENGTH + 1)}]
        with patch('app.api.get_tts_provider') as factory:
            for body in cases:
                with self.subTest(body=body):
                    response = self.client.post('/api/v1/audio/speak', json=body)
                    self.assertEqual(response.status_code, 422)
                    self.assertTrue(any(e['loc'] == ['body', 'text']
                                        for e in response.json()['detail']))
            factory.assert_not_called()

    def test_maximum_length_after_trimming_is_accepted(self):
        text = 'x' * MAX_NARRATION_TEXT_LENGTH
        response = self.client.post('/api/v1/audio/speak', json={'text': ' ' + text + ' '})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.create.call_args.kwargs['text'], text)

    def test_provider_failure_is_clean_503(self):
        self.create.side_effect = RuntimeError('secret-key-provider-details')
        self.assert_unavailable(self.client.post('/api/v1/audio/speak', json={'text': 'Text'}))

    def test_provider_initialization_failure_is_clean_503(self):
        with patch('app.api.get_tts_provider', side_effect=RuntimeError('secret-details')):
            self.assert_unavailable(self.client.post('/api/v1/audio/speak', json={'text': 'Text'}))

    def test_client_initialization_failure_is_clean_503(self):
        self.elevenlabs.side_effect = RuntimeError('secret-key-provider-details')
        self.assert_unavailable(self.client.post('/api/v1/audio/speak', json={'text': 'Text'}))

    def test_missing_or_blank_api_key_is_clean_503(self):
        for value in (None, '', ' \t'):
            with self.subTest(value=value):
                if value is None:
                    os.environ.pop('ELEVENLABS_API_KEY', None)
                else:
                    os.environ['ELEVENLABS_API_KEY'] = value
                self.assert_unavailable(self.client.post('/api/v1/audio/speak', json={'text': 'Text'}))
        self.elevenlabs.assert_not_called()

    def test_missing_or_invalid_default_voice_is_clean_503(self):
        for value in (None, '', '  ', 'alloy', '../invalid'):
            with self.subTest(value=value):
                if value is None:
                    os.environ.pop('ELEVENLABS_VOICE_ID', None)
                else:
                    os.environ['ELEVENLABS_VOICE_ID'] = value
                self.assert_unavailable(self.client.post('/api/v1/audio/speak', json={'text': 'Text'}))
        self.elevenlabs.assert_not_called()

    def test_invalid_explicit_voice_is_clean_503(self):
        for voice in ('', '  ', '../invalid', 'voice?query=value'):
            with self.subTest(voice=voice):
                self.assert_unavailable(self.client.post('/api/v1/audio/speak',
                                                        json={'text': 'Text', 'voice': voice}))
        self.elevenlabs.assert_not_called()

    def test_audio_iteration_failure_is_clean_503(self):
        def broken_audio():
            yield b'partial audio'
            raise RuntimeError('secret-key-provider-details')
        self.create.return_value = broken_audio()
        self.assert_unavailable(self.client.post('/api/v1/audio/speak', json={'text': 'Text'}))

    def test_empty_provider_audio_is_clean_503(self):
        self.create.return_value = iter([])
        self.assert_unavailable(self.client.post('/api/v1/audio/speak', json={'text': 'Text'}))
