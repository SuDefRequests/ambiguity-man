"""OCR contracts independent of the original archive exports."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Metadata(StrictModel):
    title: str | None = Field(None, max_length=500)
    source: str | None = Field(None, max_length=500)
    collection: str | None = Field(None, max_length=500)
    language: Literal['en', 'eng', 'hi', 'hin', 'mr', 'mar'] | None = None
    date: str | None = Field(None, max_length=100)
    volume: int | None = Field(None, ge=1)
    author: str | None = Field(None, max_length=500)
    notes: str | None = Field(None, max_length=10000)

class ReviewedPage(StrictModel):
    page: int = Field(ge=1)
    text: str = Field(max_length=500000)

class OCRPage(ReviewedPage):
    confidence: float | None = Field(None, ge=0, le=1)

class IngestRequest(StrictModel):
    document_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    metadata: Metadata
    pages: list[ReviewedPage] = Field(min_length=1, max_length=100)

    @model_validator(mode='after')
    def unique_pages(self):
        if len({p.page for p in self.pages}) != len(self.pages):
            raise ValueError('Duplicate page numbers')
        return self

class OCRPassage(BaseModel):
    passage_id: str
    document_id: str
    archive_type: Literal['ocr'] = 'ocr'
    source: str | None = None
    title: str | None = None
    page: int
    volume: int | None = None
    url: str | None = None
    language: str
    ocr_confidence: float | None = None
    text: str
    provenance: dict
    metadata: Metadata

class OCRSearchHit(OCRPassage):
    snippet: str
    relevance_score: float | None = None
