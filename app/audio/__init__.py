"""Archive audio provider selection; initialization stays lazy."""
from app.audio.provider import TTSProvider


def get_tts_provider() -> TTSProvider:
    from app.audio.elevenlabs_tts import ElevenLabsTTSProvider

    return ElevenLabsTTSProvider()
