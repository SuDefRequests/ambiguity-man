"""ElevenLabs narration using the application's dotenv-loaded environment."""
import os
import re

import httpx
from elevenlabs.client import ElevenLabs


class ElevenLabsTTSProvider:
    def synthesize(self, text: str, voice: str, language: str) -> bytes:
        api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        if not api_key:
            raise ValueError("TTS API key is not configured")
        voice_id = (os.getenv("ELEVENLABS_VOICE_ID", "") if voice == "alloy" else voice).strip()
        # Treat explicit values as opaque IDs, never URL paths or voice names.
        # The provider validates whether the ID exists and is accessible.
        if voice_id == "alloy" or not re.fullmatch(r"[A-Za-z0-9_-]+", voice_id):
            raise ValueError("TTS voice ID is missing or invalid")

        # This model does not require language_code. Keep language as metadata
        # and send the original text without translation or added instructions.
        with httpx.Client(timeout=30.0) as transport:
            client = ElevenLabs(api_key=api_key, timeout=30.0, httpx_client=transport)
            audio = b"".join(client.text_to_speech.convert(
                voice_id=voice_id, text=text, model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128", request_options={"max_retries": 0},
            ))
        if not audio:
            raise ValueError("TTS returned no audio")
        return audio
