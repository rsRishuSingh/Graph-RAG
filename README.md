# GRAPH RAG: Retrieval-Augmented Generation (RAG) Pipeline

A scalable RAG system that processes thousands of pages of financial and technical documents to deliver accurate, context-aware answers via a conversational chat interface.

---

## Features

* **Hybrid Retrieval:** BM25 (lexical), ChromaDB (vector), and Neo4j (graph) retrievers combined in an ensemble.
* **Semantic Chunking:** Recursive and embedding-based splitting for coherent context windows.
* **Chainlit UI:** Real-time chat deployment with persistent session management.
* **Groq LLM Integration:** Low‑temperature, context‑driven generation ensuring factual consistency.

---

## Search Components

### LLM (Groq)

* Utilizes the Groq `qwen3-32b` model with a low-temperature setting (e.g., `temperature=0.2`) to generate precise, context‑grounded responses.
* Ingests only the top-k retrieved chunks as context, preventing hallucinations and ensuring factual consistency.

### Document Chunking

* **Recursive Splitter:** Breaks documents into fixed-size segments (default `chunk_size=500`, `chunk_overlap=100`) based on hierarchical separators (`"\n\n"`, `"\n"`, `.`, `" "`).
* **Semantic Chunker:** Further refines segments using embedding-based clustering to group semantically cohesive sentences, improving retrieval relevance.

### Embeddings

* **HuggingFaceEmbeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` to convert text chunks into dense vector representations.
* Supports storage in both ChromaDB (local vector index) and Neo4jVector (graph‑based index) for diverse retrieval strategies.

### Retrievers

* **BM25 (Lexical):** Fast term‑matching retrieval using `BM25Retriever` for keyword-based lookups.
* **ChromaDB (Vector):** Approximate nearest neighbor search over embeddings for semantic similarity queries.
* **Neo4jVector (Graph):** Leverages a property graph in Neo4j to enhance context discovery via connected document embeddings.
* **Ensemble Retriever:** Combines scores from all three retrievers with equal weights (`[1,1,1]`) to balance lexical and semantic relevance.

### Interface

* **Chainlit Chat UI:** Web‑based chat server that initializes the collection, persists sessions, and handles user queries asynchronously.
* Provides feedback on startup (e.g., ⚠️ no documents, ✅ collection loaded) and streams generated answers in real time.

---
## ![Output](https://github.com/user-attachments/assets/30f87ec4-48dd-4b3b-938c-eba12a1bd463)

## Installation

1. **Clone & Setup**

   ```bash
   git clone https://github.com/rsRishuSingh/Graph-RAG.git
   cd Graph-RAG
   python -m venv venv && source venv/bin/activate
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Create a `.env` file with:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL_NAME=qwen/qwen3-32b
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_INDEX=LangChainDocs
```

Update constants in code as needed (`EMBED_MODEL_NAME`, `COLLECTION_NAME`, `CHROMA_DB_PATH`).

---

## Usage

1. **Ingest Documents** (offline):

   ```bash
   python ingest_docs.py  # Splits, chunks, and saves to all_docs.json
   ```

2. **Start Chat Server**:

   ```bash
   chainlit run app.py
   ```

3. **Interact:** open [http://localhost:8000](http://localhost:8000), upload your documents, and start querying.

---

## Impact & Relevance

1. **Accelerates Research:** Queries multi‑format corpora in seconds.
2. **Improves Accuracy:** Semantic chunking + ensemble retrieval reduces irrelevant matches.
3. **Democratizes AI:** Intuitive chat UI for enterprise users without ML expertise.

---
