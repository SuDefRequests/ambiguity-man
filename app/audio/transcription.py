"""ElevenLabs Scribe v2 speech-to-text provider."""

import os
from io import BytesIO

from elevenlabs.client import ElevenLabs


SUPPORTED_LANGUAGES = {
    "eng": "en",
    "hin": "hi",
    "mar": "mr",
}


def transcribe_audio(
    audio: bytes,
    language: str | None = None,
) -> dict[str, str | float | None]:
    api_key = os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured")

    client = ElevenLabs(api_key=api_key)

    language_code = None

    if language:
        language_code = {
            "en": "eng",
            "hi": "hin",
            "mr": "mar",
        }.get(language)

    result = client.speech_to_text.convert(
        file=BytesIO(audio),
        model_id="scribe_v2",
        language_code=language_code,
        tag_audio_events=False,
        diarize=False,
    )

    detected_code = getattr(result, "language_code", None)

    detected_language = SUPPORTED_LANGUAGES.get(
        detected_code,
        detected_code,
    )

    return {
        "text": result.text.strip(),
        "language": detected_language,
        "language_probability": getattr(
            result,
            "language_probability",
            None,
        ),
    }