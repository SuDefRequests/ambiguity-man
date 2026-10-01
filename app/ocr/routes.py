import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from app.ocr import service
from app.ocr.storage import OCRStorage
from app.ocr.models import IngestRequest
from app.ocr.ingestion import build_passages, upsert_passages

router = APIRouter(prefix='/api/v1/ocr', tags=['ocr'])
_ingestion_lock = Lock()

def get_storage():
    return OCRStorage()

@router.post('/preview')
def preview(file: UploadFile = File(...), language: str = Form('en')):
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in service.EXTENSIONS:
        raise HTTPException(400, 'Unsupported file extension')
    if language not in service.LANGUAGES:
        raise HTTPException(422, 'Unsupported OCR language')
    content = file.file.read(service.MAX_BYTES + 1)
    if not content:
        raise HTTPException(400, 'Empty file')
    if len(content) > service.MAX_BYTES:
        raise HTTPException(413, 'Upload exceeds 25 MiB')
    try:
        result = service.process_document(content, suffix, language)
    except service.InvalidDocument as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, 'OCR engine unavailable') from exc
    document_id = uuid.uuid4().hex
    result.update(document_id=document_id, filename=Path(file.filename).name,
                  sha256=hashlib.sha256(content).hexdigest(), language=service.LANGUAGES[language],
                  processing_timestamp=datetime.now(timezone.utc).isoformat(), status='completed')
    storage = get_storage()
    storage.save_original(document_id, suffix, content)
    storage.write_json('staged', document_id, result)
    return result

@router.post('/ingest')
def ingest(request: IngestRequest):
    from app.api import get_collection
    storage = get_storage()
    with _ingestion_lock:
        try:
            staged = storage.read_json('staged', request.document_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, 'Staged OCR document not found') from exc
        original = storage.path('originals', request.document_id, Path(staged['filename']).suffix.lower())
        if not original.is_file():
            raise HTTPException(409, 'Original OCR source is missing')
        reviewed = request.model_dump()
        try:
            existing = storage.read_json('approved', request.document_id)
        except FileNotFoundError:
            existing = None
        if existing:
            if existing['reviewed'] != reviewed:
                raise HTTPException(409, 'Approved document is immutable; submit a new preview for revisions')
            return {'document_id': request.document_id, 'status': 'ingested', 'passage_count': len(existing['passages']), 'duplicate': True}
        try:
            passages = build_passages(request, staged)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        storage.write_json('reviewed', request.document_id, reviewed)
        try:
            upsert_passages(get_collection(), passages)
        except Exception as exc:
            raise HTTPException(503, 'OCR indexing unavailable; retry the same request') from exc
        storage.write_json('approved', request.document_id, {'reviewed': reviewed, 'passages': [p.model_dump() for p in passages]})
    return {'document_id': request.document_id, 'status': 'ingested', 'passage_count': len(passages), 'duplicate': False}
