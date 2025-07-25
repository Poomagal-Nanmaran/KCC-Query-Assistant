
🌾 **KCC Query Assistant**

An offline‑capable, local‑first AI assistant for answering agricultural queries using the Kisan Call Center (KCC) dataset with RAG (Retrieval‑Augmented Generation) and a local LLM (via Ollama).

It works completely offline after setup:

Cleans and chunks the KCC dataset

Embeds documents using paraphrase-multilingual-mpnet-base-v2

Stores embeddings in ChromaDB

Retrieves relevant context for a query

Generates answers with Ollama (e.g., Gemma, Mistral)

✨ **Features**
Offline‑capable — Runs fully on your machine

Multilingual — Supports Indian languages for queries and data

RAG Pipeline — Combines semantic search + LLM for better answers

Fallback — Warns when no relevant local context is found

Web UI — Streamlit interface for easy querying

📂 **Project Structure**
```bash
KCCQueryAssistant/
├── data/                   # Raw & preprocessed dataset
│   ├── raw/
│   └── preprocessed/
├── vectorstore/            # ChromaDB persistent storage
├── scripts/
│   ├── preprocess.py       # Clean & chunk KCC data
│   ├── embed_and_store.py  # Generate embeddings & store in ChromaDB
│   ├── rag_core.py         # Core RAG logic (retrieval + LLM)
│   └── __init__.py
├── ui/
│   └── app.py              # Streamlit frontend
├── Dockerfile              # App container
├── docker-compose.yml      # Ollama + App services
├── requirements.txt
└── README.md
```

🛠️ **Prerequisites**
Docker & Docker Compose

At least 8GB RAM for embeddings + small LLMs

(Optional) Python 3.10+ for running without Docker

🚀 **Quick Start (with Docker)**

***1. Create a repository***
mkdir KCCQueryAssistant
***2. Place the KCC dataset***
Put the raw KCC CSV inside:
data/raw/
***3. Build docker image***
bash build_docker.sh

docker compose up
This will start:

Ollama on http://localhost:11434

Streamlit UI on http://localhost:8501

***4. Pull an LLM model***
Once Ollama is running:

docker exec -it ollama ollama pull gemma
(You can also use mistral or llama3)

***5. Preprocess & embed data***
Run these inside the container:

docker exec -it kcc-query-assistant python scripts/preprocess.py
docker exec -it kcc-query-assistant python scripts/embed_and_store.py
***6. Access the Web UI***
Go to: http://localhost:8501

**Ask queries like:**

"What pest-control methods are recommended for paddy in Tamil Nadu?"

"How to manage drought stress in groundnut cultivation?"

"Common sugarcane diseases in Maharashtra?"

⚙️ ***Environment Variables***
Variable	Default	Description
PYTHONPATH	/app	Project root for imports

🖥️ **Running Without Docker**
If you don’t want Docker:

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Run Ollama locally (host machine):

ollama serve
ollama pull gemma
Preprocess & embed:

python scripts/preprocess.py
python scripts/embed_and_store.py
Run the Streamlit UI:


streamlit run ui/app.py
