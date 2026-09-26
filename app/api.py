import os
import uuid
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import PyPDF2
from gtts import gTTS
from fastapi.responses import FileResponse

# 1. Load API keys from .env
load_dotenv()

# 2. CREATE THE APP FIRST (This is what was causing the error)
app = FastAPI(title="Abhilekh Knowledge API", version="0.1.0")

# 3. Initialize Groq and ChromaDB
# Groq will automatically use the GROQ_API_KEY from your .env file
groq_client = Groq()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
emb_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_collection(
    name="abhilekh_knowledge",
    embedding_function=emb_fn
)

# 4. Import your local models
from app.models import (
    SearchRequest, QuoteRequest, QuoteResponse,
    GraphResponse, GraphNode, GraphEdge
)
from app.quote_authenticator import verify_quote
from app.graph.query import article_subgraph


# --- ENDPOINTS ---

@app.post("/search")
def search(req: SearchRequest):
    # 1. Retrieve local offline data
    results = collection.query(
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
        completion = groq_client.chat.completions.create(
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
                transcription = groq_client.audio.transcriptions.create(
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
    
    collection.upsert(
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