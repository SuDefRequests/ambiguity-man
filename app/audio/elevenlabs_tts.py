"""ElevenLabs narration with primary/fallback API credentials."""

import os
import re

import httpx
from elevenlabs.client import ElevenLabs


class ElevenLabsTTSProvider:
    def _get_voice_id(self, voice: str, fallback: bool = False) -> str:
        if voice != "alloy":
            return voice.strip()

        variable = (
            "ELEVENLABS_VOICE_ID_FALLBACK"
            if fallback
            else "ELEVENLABS_VOICE_ID"
        )

        return os.getenv(variable, "").strip()

    def _synthesize_with_key(
        self,
        api_key: str,
        voice_id: str,
        text: str,
    ) -> bytes:
        with httpx.Client(timeout=30.0) as transport:
            client = ElevenLabs(
                api_key=api_key,
                timeout=30.0,
                httpx_client=transport,
            )

            audio = b"".join(
                client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                    request_options={"max_retries": 0},
                )
            )

        if not audio:
            raise ValueError("TTS returned no audio")

        return audio

    def synthesize(
        self,
        text: str,
        voice: str,
        language: str,
    ) -> bytes:
        primary_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        fallback_key = os.getenv(
            "ELEVENLABS_API_KEY_FALLBACK",
            "",
        ).strip()

        primary_voice = self._get_voice_id(
            voice,
            fallback=False,
        )
        fallback_voice = self._get_voice_id(
            voice,
            fallback=True,
        )

        # Validate primary configuration.
        if primary_key:
            if (
                primary_voice == "alloy"
                or not re.fullmatch(
                    r"[A-Za-z0-9_-]+",
                    primary_voice,
                )
            ):
                raise ValueError(
                    "Primary TTS voice ID is missing or invalid"
                )

        # If no primary key exists, use fallback directly.
        if not primary_key:
            primary_key = fallback_key
            primary_voice = fallback_voice
            fallback_key = ""

        if not primary_key:
            raise ValueError("No TTS API key is configured")

        try:
            return self._synthesize_with_key(
                primary_key,
                primary_voice,
                text,
            )

        except Exception as primary_error:
            # Try the fallback credentials only when configured.
            if not fallback_key:
                raise primary_error

            if (
                fallback_voice == "alloy"
                or not re.fullmatch(
                    r"[A-Za-z0-9_-]+",
                    fallback_voice,
                )
            ):
                raise ValueError(
                    "Fallback TTS voice ID is missing or invalid"
                ) from primary_error

            try:
                return self._synthesize_with_key(
                    fallback_key,
                    fallback_voice,
                    text,
                )
            except Exception:
                # Preserve the original failure if both providers fail.
                raise primary_error