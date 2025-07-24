# scripts/rag_core.py

import os
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import requests
import json
# ---- Config ----
CHROMA_DB_DIR = "/app/vectorstore/kcc_chromadb"
COLLECTION_NAME = "kcc_queries"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
OLLAMA_URL = "http://ollama:11434/api/generate"
MODEL_NAME = "gemma"
TOP_K = 5

# ---- Load Embedding Model ----
embedder = SentenceTransformer(EMBED_MODEL)
chroma_client = PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)

def retrieve_context(query, top_k=TOP_K):
    query_embedding = embedder.encode([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    if results:
        print("output from chroma")
    return results

def construct_prompt(query, context_chunks):
    print("constructing prompt")
    context_text = "\n\n".join(f"- {doc}" for doc in context_chunks)
    return (
        f"You are an agricultural expert answering farmer queries using the following information:\n\n"
        f"{context_text}\n\n"
        f"Question:\n{query}\n\n"
        f"Answer:"
    )

def query_ollama(prompt):
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}
    response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()["response"]

def rag_query_pipeline(query):
    results = retrieve_context(query)
    docs = results.get("documents", [[]])[0]
    if not docs:
        return None, []

    prompt = construct_prompt(query, docs)
    print("querying ollama",prompt)
    response = query_ollama(prompt)
    return response, docs