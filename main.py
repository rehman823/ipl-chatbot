"""FastAPI backend: POST /chat and GET /health.

Run:  uvicorn main:app --port 8000

Questions about seasons after the dataset, or about "now" (latest, today, live...), are answered
from live Tavily web results. Everything else is answered from the local ChromaDB dataset.
"""
from datetime import date

import chromadb
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import web_search
from ollama_client import embed
from web_search import WebSearchError

app = FastAPI(title="IPL Chatbot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
collection = client.get_or_create_collection(
    config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)

SYSTEM_PROMPT = (
    "You answer questions about IPL cricket using ONLY the sources provided below. "
    "Sources marked [Dataset] are historical match and ball-by-ball records. "
    "Sources marked [Web] are live web search results; when you use one, mention its date or site. "
    "If the answer is not in the sources, reply exactly: "
    "\"I couldn't find that in the available sources.\" Do not use outside knowledge or guess."
)


class ChatRequest(BaseModel):
    question: str


def dataset_sources(question):
    """Embed the question and fetch the most similar dataset chunks."""
    total = collection.count()
    found = collection.query(
        query_embeddings=embed([question], prefix=config.QUERY_PREFIX),
        n_results=min(config.TOP_K, total),
        include=["documents", "metadatas"],
    )
    return [
        {"type": "dataset", "source": m["source"], "match": m["match"],
         "chunk_index": m["chunk_index"], "text": d}
        for d, m in zip(found["documents"][0], found["metadatas"][0])
    ]


def web_sources(question):
    return [
        {"type": "web", "title": r["title"], "url": r["url"],
         "published_date": r["published_date"], "text": r["content"]}
        for r in web_search.search_web(question)
    ]


def build_context(sources):
    blocks = []
    for i, s in enumerate(sources, 1):
        if s["type"] == "web":
            when = s["published_date"] or "date unknown"
            blocks.append(f"[Web {i}] {s['title']} ({when}) {s['url']}\n{s['text']}")
        else:
            blocks.append(f"[Dataset {i}]\n{s['text']}")
    return "\n\n---\n\n".join(blocks)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": collection.count(),
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "live_search": web_search.is_enabled(),
    }


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    sources, notice = [], None

    # 1. Live questions go to Tavily first
    if web_search.needs_live(question):
        if not web_search.is_enabled():
            notice = "Live search is off (TAVILY_API_KEY is not set), so only the dataset was searched."
        else:
            try:
                sources = web_sources(question)
            except WebSearchError as e:
                notice = f"Live search failed ({e}), so only the dataset was searched."
            if not sources and not notice:
                notice = "Live search returned no results, so only the dataset was searched."

    try:
        # 2. Otherwise (or as a fallback) use the local dataset
        if not sources:
            if collection.count() == 0:
                raise HTTPException(status_code=503, detail="The vector store is empty. Run ingest.py first.")
            sources = dataset_sources(question)

        # 3. Ask the LLM with the labelled sources
        r = requests.post(
            f"{config.OLLAMA_URL}/api/chat",
            json={
                "model": config.LLM_MODEL,
                "stream": False,
                "options": {"temperature": 0.2, "num_ctx": config.LLM_NUM_CTX},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":
                        f"Today's date: {date.today().isoformat()}\n\n"
                        f"Sources:\n{build_context(sources)}\n\nQuestion: {question}"},
                ],
            },
            timeout=300,
        )
        r.raise_for_status()
        answer = r.json()["message"]["content"].strip()
    except (requests.RequestException, RuntimeError, KeyError) as e:
        raise HTTPException(status_code=502, detail=f"Ollama problem: {e}")

    mode = "web" if sources and sources[0]["type"] == "web" else "dataset"
    return {"answer": answer, "mode": mode, "notice": notice, "sources": sources}
