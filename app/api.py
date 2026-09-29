import os
import uuid
import tempfile
from functools import lru_cache
from typing import Literal
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from dotenv import load_dotenv
from fastapi.responses import FileResponse, Response
from pydantic import StringConstraints
from typing import Annotated



# 1. Load API keys from .env
load_dotenv()

# 2. CREATE THE APP FIRST (This is what was causing the error)
app = FastAPI(title="Abhilekh Knowledge API", version="0.1.0")

# Optional services are initialized only by the routes that use them.
@lru_cache(maxsize=1)
def get_groq_client():
    from groq import Groq
    return Groq()


@lru_cache(maxsize=1)
def get_collection():
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection(
        name="abhilekh_knowledge",
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )


# 4. Import your local models
from app.models import (
    ArchiveAdjacencyResponse, ArchiveSearchResponse, ArchiveBrowseResponse, ArchivePassage, SearchRequest, QuoteRequest, QuoteResponse,
    GraphResponse, GraphNode, GraphEdge, AskRequest, AskResponse, SpeakRequest, TranscriptionResponse
)
from app.quote_authenticator import verify_quote
from app.archive import browse_passages, get_passages, passage_adjacency
from app.audio import get_tts_provider


# --- ENDPOINTS ---

@app.get("/api/v1/passages", response_model=ArchiveBrowseResponse)
def list_passages(
    archive: Literal["all", "baws", "cad"] = "all",
    volume: int | None = Query(None, ge=1, le=5),
    offset: int = Query(0, ge=0),
    limit: int = Query(8, ge=1, le=20),
):
    """Browse in original export order; all combines BAWS followed by CAD.

    Archive and volume filters retain source order and are applied before
    offset/limit. Total counts all matching passages before pagination.
    """
    return browse_passages(archive, volume, offset, limit)


@app.get("/api/v1/passages/{passage_id}", response_model=ArchivePassage)
def passage_by_id(passage_id: str):
    passage = get_passages().get(passage_id)
    if passage is None:
        raise HTTPException(status_code=404, detail={
            "code": "passage_not_found",
            "message": "Passage not found",
            "passage_id": passage_id,
        })
    return passage


@app.get("/api/v1/passages/{passage_id}/adjacency", response_model=ArchiveAdjacencyResponse)
def get_passage_adjacency(passage_id: str):
    """Return source neighbors within a BAWS volume or CAD volume/session."""
    adjacency = passage_adjacency(passage_id)
    if adjacency is None:
        raise HTTPException(status_code=404, detail={
            "code": "passage_not_found",
            "message": "Passage not found",
            "passage_id": passage_id,
        })
    return adjacency


@app.get("/api/v1/search", response_model=ArchiveSearchResponse)
def search_passages(
    q: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1), Query()],
    archive: Literal["all", "baws", "cad"] = "all",
    volume: int | None = Query(None, ge=1, le=5),
    limit: int = Query(8, ge=1, le=20),
):
    """Semantic passage retrieval. total_results counts returned hydrated hits.

    No generated answer or corpus-wide match count is provided. Empty/blank
    queries are invalid; an empty retrieval returns results=[] and total_results=0.
    """
    from app.passage_search import retrieve_passages
    try:
        collection = get_collection()
    except Exception as exc:
        raise HTTPException(503, detail={
            "code": "search_unavailable", "message": "Archive search is unavailable",
        }) from exc
    return retrieve_passages(collection, q, archive, volume, limit)


@app.post("/api/v1/ask", response_model=AskResponse)
def ask_archive(req: AskRequest):
    from app.ask import build_messages
    from app.passage_search import retrieve_passages
    from app.query_translation import translate_query_for_retrieval
    try:
        collection = get_collection()
    except Exception as exc:
        raise HTTPException(503, detail={
            "code": "search_unavailable", "message": "Archive search is unavailable",
        }) from exc
    retrieval_question = translate_query_for_retrieval(
        req.question,
        req.language or "en",
    )

    print(
        f"🔎 Retrieval query: {retrieval_question!r}",
        flush=True,
    )

    retrieved = retrieve_passages(
        collection,
        retrieval_question,
        req.archive,
        req.volume,
        req.top_k,
    )
    
    sources = [ArchivePassage.model_validate(hit.model_dump()) for hit in retrieved.results]
    if not sources:
        return AskResponse(question=req.question,
                           answer="No relevant archive evidence was found for this question.", sources=[])
    try:
        completion = get_groq_client().chat.completions.create(
            model="qwen/qwen3.8-27b",  # Same model as the existing POST /search.
            messages=build_messages(req.question, sources, req.language or "en"),
            temperature=0.2,
        )
        answer = completion.choices[0].message.content
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Empty generation")
    except Exception as exc:
        print("🔥 GROQ GENERATION ERROR:", repr(exc), flush=True)
        raise HTTPException(503, detail={
            "code": "generation_unavailable", "message": "Archive answer generation is unavailable",
        }) from exc
    return AskResponse(question=req.question, answer=answer.strip(), sources=sources)


@app.post("/api/v1/audio/speak", response_class=Response,
          responses={200: {"content": {"audio/mpeg": {"schema": {"type": "string", "format": "binary"}}}}})
def speak_archive(req: SpeakRequest):
    try:
        provider = get_tts_provider()
        audio = provider.synthesize(text=req.text, voice=req.voice, language=req.language)
    except Exception as exc:
        raise HTTPException(503, detail={
            "code": "tts_unavailable",
            "message": "Archive narration is currently unavailable.",
        }) from exc
    return Response(content=audio, media_type="audio/mpeg", headers={
        "Content-Disposition": 'attachment; filename="archive-narration.mp3"',
    })

@app.post(
    "/api/v1/audio/transcribe",
    response_model=TranscriptionResponse,
)
async def transcribe_archive_audio(
    file: UploadFile = File(...),
    language: Literal["en", "hi", "mr"] | None = None,
):
    from app.audio.transcription import transcribe_audio

    try:
        audio = await file.read()

        if not audio:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "empty_audio",
                    "message": "The uploaded audio file is empty.",
                },
            )

        result = transcribe_audio(
            audio=audio,
            language=language,
        )

        return TranscriptionResponse(**result)

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "stt_unavailable",
                "message": "Archive speech recognition is currently unavailable.",
            },
        ) from exc




@app.post("/search")
def search(req: SearchRequest):
    # 1. Retrieve local offline data
    results = get_collection().query(
        query_texts=[req.query],
        n_results=req.top_k
    )
    
    if results["documents"] and len(results["documents"][0]) > 0:
        context_text = "\n\n---\n\n".join(results["documents"][0])
        sources = results["documents"][0]
    else:
        context_text = "No relevant historical records found in the database."
        sources = []

    # 2. Fast Cloud Generation via Groq
    prompt = f"""You are a historical legal assistant. Answer the user's question using ONLY the following retrieved documents. If the answer is not in the documents, say "I cannot find this in the archives."

Documents:
{context_text}

Question: {req.query}"""

    # 3. CRASH-PROOF WRAPPER + NEW MODEL
    try:
        completion = get_groq_client().chat.completions.create(
            model="qwen/qwen3.8-27b", # <-- Updated active model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, 
        )
        ai_answer = completion.choices[0].message.content
    except Exception as e:
        ai_answer = f"⚠️ AI Generation Error: {str(e)}"

    return {
        "query": req.query,
        "answer": ai_answer,
        "sources": sources
    }

@app.post("/speak")
def generate_audio(text_to_speak: str):
    if not text_to_speak.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    # Generate MP3 audio from the text
    from gtts import gTTS
    tts = gTTS(text=text_to_speak, lang='en', slow=False)
    output_path = f"accessibility_audio_{uuid.uuid4().hex[:6]}.mp3"
    tts.save(output_path)
    
    # Return the audio file directly to the frontend
    return FileResponse(output_path, media_type="audio/mpeg", filename="response.mp3")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = file.filename.split('.')[-1].lower()
    full_text = ""
    
    # Extract PDF Text
    if ext == "pdf":
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(file.file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"
                
    # Transcribe Audio/Video via Groq Whisper
    elif ext in ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            
        try:
            with open(tmp_path, "rb") as audio_file:
                transcription = get_groq_client().audio.transcriptions.create(
                    file=(file.filename, audio_file.read()),
                    model="whisper-large-v3",
                )
            full_text = transcription.text
        finally:
            os.remove(tmp_path)
            
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text.")

    # Chunk and Insert into ChromaDB
    chunk_size = 1000
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    
    chunk_ids = [f"{file.filename}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
    metadatas = [{"source": file.filename, "archive_type": "dynamic_upload"} for _ in chunks]
    
    get_collection().upsert(
        documents=chunks,
        metadatas=metadatas,
        ids=chunk_ids
    )
    
    return {
        "status": "success", 
        "filename": file.filename, 
        "chunks_added": len(chunks),
        "message": f"Successfully learned from {file.filename}!"
    }


@app.post("/verify-quote", response_model=QuoteResponse)
def verify(req: QuoteRequest):
    if not req.quote.strip():
        raise HTTPException(status_code=400, detail="quote must not be empty")
    result = verify_quote(req.quote)
    return QuoteResponse(**result)


@app.get("/graph/article/{number}", response_model=GraphResponse)
def graph_article(number: str):
    from app.graph.query import article_subgraph
    sub = article_subgraph(number)
    return GraphResponse(
        nodes=[GraphNode(**n) for n in sub["nodes"]],
        edges=[
            GraphEdge(source=e["source"], target=e["target"],
                      type=e["type"], passage_id=e.get("passage_id"))
            for e in sub["edges"]
        ],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
