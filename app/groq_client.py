import os
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

RETRYABLE_STATUS_CODES = {401, 429, 500, 502, 503, 504}
RETRYABLE_EXCEPTION_NAMES = {"APIConnectionError", "APITimeoutError"}


@lru_cache(maxsize=1)
def get_groq_clients():
    clients = []

    primary = os.getenv("GROQ_API_KEY")
    fallback = os.getenv("GROQ_API_KEY_FALLBACK")

    if primary:
        clients.append(Groq(api_key=primary))

    if fallback:
        clients.append(Groq(api_key=fallback))

    if not clients:
        raise RuntimeError("No Groq API key configured")

    return tuple(clients)


class _GroqFallbackProxy:
    def __init__(self, clients, path=()):
        self._clients = clients
        self._path = path

    def __getattr__(self, name):
        return _GroqFallbackProxy(
            self._clients,
            self._path + (name,),
        )

    def __call__(self, *args, **kwargs):
        last_exc = None

        for index, client in enumerate(self._clients):
            target = client

            for part in self._path:
                target = getattr(target, part)

            try:
                return target(*args, **kwargs)

            except Exception as exc:
                last_exc = exc

                status = getattr(exc, "status_code", None)
                exception_name = type(exc).__name__

                retryable = (
                    status in RETRYABLE_STATUS_CODES
                    or exception_name in RETRYABLE_EXCEPTION_NAMES
                )

                if index == len(self._clients) - 1 or not retryable:
                    raise

                print(
                    f"⚠️ GROQ KEY {index + 1} failed "
                    f"({exception_name}, status={status}). "
                    f"Trying fallback key...",
                    flush=True,
                )

        raise last_exc


@lru_cache(maxsize=1)
def get_groq_client():
    return _GroqFallbackProxy(get_groq_clients())
