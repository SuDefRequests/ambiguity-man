import os
from dotenv import load_dotenv
from groq import Groq
from app.core.indexer import search_archive

# Load environment variables from .env file
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set. Please check your .env file.")

groq_client = Groq(api_key=groq_api_key)

def generate_grounded_answer(query: str, top_k: int = 3):
    # 1. RETRIEVE (Vector Search via ChromaDB)
    chunks = search_archive(query_text=query, n_results=top_k)
    
    if not chunks:
        return {
            "query": query,
            "answer": "No archival records were found regarding this specific topic in the database.",
            "citations": []
        }
    
    # 2. AUGMENT (Assemble verified context with page numbers)
    context_blocks = []
    citations = []
    for c in chunks:
        context_blocks.append(
            f"[Source: {c['source']}, Page: {c['page']}]\n\"{c['snippet']}\""
        )
        citations.append({
            "source": c["source"],
            "page": c["page"],
            "snippet": c["snippet"][:180] + "..."
        })
    
    joined_context = "\n\n".join(context_blocks)
    
    system_instruction = (
        "You are the AI Research Assistant for the Dr. Ambedkar Heritage Archive at the "
        "Dr. Ambedkar International Centre. Answer the visitor's question strictly and exclusively "
        "using the provided archival excerpts. If the excerpts do not contain enough facts to answer, "
        "state that the archive has no documented record on this query. Always cite the relevant page numbers "
        "in your answer. Keep your response direct, factual, and within 3 to 4 sentences."
    )
    
    user_prompt = f"""Archival Context:
{joined_context}

Visitor Question: {query}
"""

    # 3. GENERATE (Using Llama 3.3 on Groq)
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=400
    )
    
    answer_text = chat_completion.choices[0].message.content.strip()
    
    return {
        "query": query,
        "answer": answer_text,
        "citations": citations
    }