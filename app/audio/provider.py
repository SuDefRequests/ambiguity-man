"""Provider-independent contract for archive narration."""
from typing import Protocol


class TTSProvider(Protocol):
    def synthesize(self, text: str, voice: str, language: str) -> bytes:
        """Return MP3 bytes, or raise if narration is unavailable."""
        ...
