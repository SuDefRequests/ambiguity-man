"""Deterministic, append-only approved ingestion into the existing collection."""
import hashlib
import re
from app.ocr.models import OCRPassage


def chunk_text(text, size=1200):
    chunks, current = [], ''
    for paragraph in re.split(r'\n\s*\n', text.strip()):
        for start in range(0, len(paragraph), size):
            part = paragraph[start:start + size].strip()
            if not part:
                continue
            if current and len(current) + len(part) + 2 > size:
                chunks.append(current)
                current = ''
            current = (current + '\n\n' + part).strip()
    if current:
        chunks.append(current)
    return chunks


def build_passages(request, staged):
    pages = {p['page']: p for p in staged['pages']}
    if set(p.page for p in request.pages) != set(pages):
        raise ValueError('Reviewed pages must match all staged page numbers')
    passages = []
    for page in sorted(request.pages, key=lambda p: p.page):
        provenance = {key: staged[key] for key in ('filename', 'sha256', 'processing_timestamp', 'ocr_engine', 'ocr_engine_version')}
        provenance.update(original_filename=staged['filename'], ocr_language=staged['language'], page=page.page, ocr_confidence=pages[page.page]['confidence'])
        for index, text in enumerate(chunk_text(page.text)):
            identity = f'{request.document_id}:{page.page}:{index}'
            passages.append(OCRPassage(
                passage_id='ocr_' + hashlib.sha256(identity.encode()).hexdigest(),
                document_id=request.document_id, source=request.metadata.source,
                title=request.metadata.title, page=page.page, volume=request.metadata.volume,
                language=request.metadata.language or staged['language'],
                ocr_confidence=pages[page.page]['confidence'], text=text,
                provenance=provenance, metadata=request.metadata,
            ))
    if not passages:
        raise ValueError('Reviewed document contains no text')
    return passages


def upsert_passages(collection, passages):
    for start in range(0, len(passages), 100):
        batch = passages[start:start + 100]
        metadatas = []
        for passage in batch:
            metadata = {k: v for k, v in passage.model_dump().items() if k not in ('metadata', 'provenance', 'text', 'url') and v is not None}
            metadata.update({k: v for k, v in passage.metadata.model_dump().items() if v is not None})
            metadata.update({k: v for k, v in passage.provenance.items() if v is not None})
            metadatas.append(metadata)
        collection.upsert(ids=[p.passage_id for p in batch], documents=[p.text for p in batch], metadatas=metadatas)
