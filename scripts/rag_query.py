import json
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import requests
import os

# ---- Config ----
CHROMA_DB_DIR = "/app/vectorstore/kcc_chromadb"
COLLECTION_NAME = "kcc_queries"
#EMBED_MODEL = "all-mpnet-base-v2"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"#"all-mpnet-base-v2"  # Or "BAAI/bge-large-en"

#OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_URL = os.getenv("OLLAMA_API", "http://localhost:11434")

MODEL_NAME = "gemma"  # Or any model you pulled in Ollama
TOP_K = 5
RELEVANCE_THRESHOLD = 0.5  # Optional (semantic scores from Chroma are not exposed directly)

# ---- Load embedding model ----
embedder = SentenceTransformer(EMBED_MODEL)

# ---- ChromaDB client ----
chroma_client = PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)

def retrieve_context(query):
    query_embedding = embedder.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )
    return results

def construct_prompt(query, context_chunks):
    context_text = "\n\n".join(
        f"- {doc}" for doc in context_chunks
    )
    return (
        f"You are an agricultural expert answering farmer queries based only on the given KCC knowledge base below.\n\n"
        f"### Knowledge Base:\n{context_text}\n\n"
        f"### Question:\n{query}\n\n"
        f"### Answer:"
    )

def query_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]

def rag_query_pipeline(query):
    results = retrieve_context(query)
    context_docs = results.get("documents", [[]])[0]  # list of retrieved docs
    if not context_docs:
        return "No relevant local KCC context found. Try online search."

    prompt = construct_prompt(query, context_docs)
    response = query_ollama(prompt)
    return response

if __name__ == "__main__":
    print("KCC Query Assistant (RAG + Ollama)")
    while True:
        user_input = input("\nAsk your query (or 'exit'): ").strip()
        if user_input.lower() == "exit":
            break
        print("\n🤖 Generating response...\n")
        answer = rag_query_pipeline(user_input)
        print(f"\n💡 Answer:\n{answer}")
