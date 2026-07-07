import os
import json
import uuid
import collections
from datetime import datetime, timedelta
from typing import List, Dict, Any, Generator
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import cohere

# --- CONFIGURATION & INITIALIZATION ---
app = FastAPI(title="ISRO SATCOM NGP RAG Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
cohere_client = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

# Models
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

# Vector Store Setup
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="satcom_ngp_clauses")

# In-Memory Database Simulation for Server-Side Metrics & Sessions
SESSION_MEMORY = collections.defaultdict(list)
SERVER_TOKEN_BUDGET = collections.defaultdict(lambda: {"used": 0, "limit": 500000})
SERVER_USAGE_LOGS = []

# --- SCHEMAS ---
class QueryRequest(BaseModel):
    session_id: str
    query: str

class TrackUsageRequest(BaseModel):
    session_id: str
    endpoint: str
    model: str
    prompt_tokens: int
    completion_tokens: int

# --- CORE RAG PIPELINE ENGINE ---

def expand_query(original_query: str) -> List[str]:
    """Generates variations to expand search recall."""
    prompt = f"Generate 3 distinct search variations for the following regulatory policy query to capture synonyms and implicit meanings: '{original_query}'. Return as a clean JSON list of strings only."
    try:
        response = gemini_model.generate_content(prompt)
        variations = json.loads(response.text.strip())
        if isinstance(variations, list):
            return [original_query] + variations
    except Exception:
        pass
    return [original_query]

def hybrid_retrieve(expanded_queries: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
    """Executes dense vector searches combined across all query variations."""
    all_results = []
    seen_ids = set()
    
    for query in expanded_queries:
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        if not results or not results['documents']:
            continue
            
        for i in range(len(results['documents'][0])):
            doc_id = results['ids'][0][i]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                meta = results['metadatas'][0][i]
                all_results.append({
                    "id": doc_id,
                    "text": results['documents'][0][i],
                    "metadata": meta
                })
    return all_results

def apply_cohere_rerank(query: str, documents: List[Dict[str, Any]], top_n: int = 4) -> List[Dict[str, Any]]:
    """Reranks retrieved candidates using Cohere's ranking endpoint."""
    if not cohere_client or not documents:
        return documents[:top_n]
    
    doc_texts = [doc["text"] for doc in documents]
    rerank_results = cohere_client.rerank(
        query=query,
        documents=doc_texts,
        top_n=top_n,
        model="rerank-english-v3.0"
    )
    
    final_docs = []
    for hit in rerank_results.results:
        final_docs.append(documents[hit.index])
    return final_docs

def agentic_context_resolution(parent_text: str) -> str:
    """Tier 2 resolution handling cross-references structurally."""
    prompt = f"Analyze the following regulatory policy text snippet. Identify if it explicitly cross-references another section or article to resolve its meaning (e.g., 'as defined in Section 4.2'). If yes, output only the section path key. If no external reference is mandatory to answer general queries, write 'NONE'.\nText:\n{parent_text}"
    try:
        ref_check = gemini_model.generate_content(prompt).text.strip()
        if "NONE" not in ref_check and len(ref_check) < 20:
            # Secondary vector search step to fetch referenced article context
            ref_embedding = embedding_model.encode(ref_check).tolist()
            sec_res = collection.query(query_embeddings=[ref_embedding], n_results=1)
            if sec_res and sec_res['documents']:
                return f"\n[Supplementary Referenced Context ({ref_check})]:\n" + sec_res['documents'][0][0]
    except Exception:
        pass
    return ""

# --- ENDPOINTS ---

@app.post("/api/usage/track")
def track_usage(payload: TrackUsageRequest):
    total = payload.prompt_tokens + payload.completion_tokens
    SERVER_TOKEN_BUDGET[payload.session_id]["used"] += total
    
    # Track metrics server-side
    SERVER_USAGE_LOGS.append({
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": payload.session_id,
        "endpoint": payload.endpoint,
        "model": payload.model,
        "prompt_tokens": payload.prompt_tokens,
        "completion_tokens": payload.completion_tokens,
        "total_tokens": total
    })
    return {"status": "tracked", "current_budget_used": SERVER_TOKEN_BUDGET[payload.session_id]["used"]}

@app.post("/api/chat/stream")
async def chat_stream(payload: QueryRequest):
    # Enforce Server-Side Token Safety Limits
    if SERVER_TOKEN_BUDGET[payload.session_id]["used"] >= SERVER_TOKEN_BUDGET[payload.session_id]["limit"]:
        raise HTTPException(status_status=429, detail="Server side global token budget exceeded.")

    def event_generator() -> Generator[str, None, None]:
        # Tier 1 & Query Expansion Execution
        queries = expand_query(payload.query)
        candidates = hybrid_retrieve(queries)
        reranked = apply_cohere_rerank(payload.query, candidates)
        
        context_blocks = []
        for doc in reranked:
            # Fetch entire context mapping or metadata paths
            path_str = f"Art: {doc['metadata'].get('article')}, Sec: {doc['metadata'].get('section')}"
            block = f"[{path_str}]\n{doc['text']}"
            # Agentic Tier 2 Loop Check
            supplement = agentic_context_resolution(doc['text'])
            block += supplement
            context_blocks.append(block)
            
        context_payload = "\n\n".join(context_blocks)
        
        # Build chat history string for context conservation
        history = SESSION_MEMORY[payload.session_id]
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])
        
        system_prompt = f"You are an expert system analyzing ISRO's SATCOM NGP Document. Answer using the validated context and chat history below. If you cannot answer using the text, clarify honestly.\n\n[Context]\n{context_payload}\n\n[History]\n{history_str}\n\nUser: {payload.query}\nAssistant:"
        
        # Stream Generation
        response_stream = gemini_model.generate_content(system_prompt, stream=True)
        full_response = ""
        
        for chunk in response_stream:
            if chunk.text:
                full_response += chunk.text
                yield f"data: {json.dumps({'text': chunk.text})}\n\n"
                
        # Append clean turn details to in-memory session stack
        SESSION_MEMORY[payload.session_id].append({"role": "user", "content": payload.query})
        SESSION_MEMORY[payload.session_id].append({"role": "assistant", "content": full_response})
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")