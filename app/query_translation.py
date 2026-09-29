from functools import lru_cache


SUPPORTED_LANGUAGES = {"en", "hi", "mr"}


@lru_cache(maxsize=1)
def get_groq_client():
    from groq import Groq
    return Groq()


def translate_query_for_retrieval(question: str, language: str) -> str:
    """Translate a Hindi/Marathi question into English for archive retrieval."""

    if language == "en":
        return question

    if language not in SUPPORTED_LANGUAGES:
        return question

    prompt = f"""Translate the following archive research question into concise English.

This translation is ONLY for semantic retrieval against an English-language
historical archive. Preserve the meaning, names, historical terms, and subject
of the question. Do not answer the question. Return only the English translation.

Question:
{question}
"""

    completion = get_groq_client().chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "system",
                "content": "You translate archival research queries accurately for semantic search.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    translated = completion.choices[0].message.content

    if not isinstance(translated, str) or not translated.strip():
        raise ValueError("Empty retrieval query translation")

    return translated.strip()
