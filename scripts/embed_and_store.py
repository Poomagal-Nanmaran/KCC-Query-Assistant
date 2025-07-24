import json
import os
from sentence_transformers import SentenceTransformer
import chromadb
#from chromadb.config import Settings
from chromadb import PersistentClient


# ---- Config ----
INPUT_JSON = "/app/data/preprocessed/kcc_cleaned_qa.json"
CHROMA_DB_DIR = "/app/vectorstore/kcc_chromadb"
COLLECTION_NAME = "kcc_queries"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"#"all-mpnet-base-v2"  # Or "BAAI/bge-large-en"

# ---- Load Data ----
def load_qa_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# ---- Embedding ----
def get_embeddings(model, texts):
    return model.encode(texts, convert_to_tensor=False, show_progress_bar=True)

# batching utility
def batch_iterable(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

# ---- Main ----
def main():
    # Load and prepare data
    qa_data = load_qa_data(INPUT_JSON)
    queries = [item["query"] for item in qa_data]
    metadatas = [item["metadata"] for item in qa_data]
    ids = [f"kcc-{i}" for i in range(len(qa_data))]

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Generating embeddings...")
    embeddings = get_embeddings(model, queries)

    # Initialize Chroma
    print("Storing in ChromaDB...")
    # chroma_client = chromadb.Client(Settings(
    #     chroma_db_impl="duckdb+parquet",
    #     persist_directory=CHROMA_DB_DIR
    # ))
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # Add to collection
    # collection.add(
    #     documents=queries,
    #     embeddings=embeddings,
    #     metadatas=metadatas,
    #     ids=ids
    # )
    BATCH_SIZE = 1000
    for q_batch, e_batch, m_batch, id_batch in zip(
        batch_iterable(queries, BATCH_SIZE),
        batch_iterable(embeddings, BATCH_SIZE),
        batch_iterable(metadatas, BATCH_SIZE),
        batch_iterable(ids, BATCH_SIZE)
    ):
        collection.add(
            documents=q_batch,
            embeddings=e_batch,
            metadatas=m_batch,
            ids=id_batch
        )


    # Persist the DB
    #chroma_client.persist()
    print(f"Stored {len(queries)} entries in ChromaDB.")

if __name__ == "__main__":
    main()
