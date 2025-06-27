import os
import json
import time
import re
import fitz  # PyMuPDF
from dotenv import load_dotenv
from typing import List

# LangChain and embedding imports
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ChromaDB imports
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction

# BM25 for keyword retrieval
from rank_bm25 import BM25Okapi

# Neo4j vector store
from langchain_neo4j import Neo4jVector

# Groq client for LLM inference
from groq import Groq

# --- Configuration ---
load_dotenv()
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "qwen/qwen3-32b"
COLLECTION_NAME = "TESLA_RAG_DOCS"
CHROMA_DB_PATH = "chromaDB/saved/"
PDF_DIR = "PDFs/"
PDF_FILES = ["TESLA"]
ALL_DOCS_JSON = "all_docs.json"

# Neo4j settings via env
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PW = os.getenv("NEO4J_PASSWORD")
NEO4J_INDEX = os.getenv("NEO4J_INDEX", "LangChainDocs")

# --- Embedding wrapper for ChromaDB ---
class LocalHFEmbedding(EmbeddingFunction):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

# --- Text Chunking ---

def recursive_split(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


def semantic_chunker(text: str, embeddings_model) -> List[str]:
    chunks = []
    for seg in recursive_split(text):
        chunker = SemanticChunker(embeddings_model)
        chunks.extend(chunker.split_text(seg))
    return chunks

# --- PDF Extraction ---

def extract_chunks_from_pdf(pdf_path: str, embeddings_model) -> List[Document]:
    docs: List[Document] = []
    pdf = fitz.open(pdf_path)
    for page_idx, page in enumerate(pdf):
        text = re.sub(r'\s+', ' ', page.get_text("text")).strip()
        if not text:
            continue
        for idx, chunk in enumerate(semantic_chunker(text, embeddings_model)):
            docs.append(Document(
                page_content=chunk,
                metadata={"page": page_idx + 1, "chunk": idx, "source": os.path.basename(pdf_path)}
            ))
    pdf.close()
    return docs

# --- Save/Load ---

def save_docs(docs: List[Document], filepath: str = ALL_DOCS_JSON) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([{"page_content": d.page_content, "metadata": d.metadata} for d in docs], f, indent=2)


def load_docs(filepath: str = ALL_DOCS_JSON) -> List[Document]:
    if not os.path.exists(filepath): return []
    with open(filepath, "r", encoding="utf-8") as f:
        arr = json.load(f)
    return [Document(page_content=a["page_content"], metadata=a["metadata"]) for a in arr]

# --- Initialize vector stores ---

def init_chroma():
    client = PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=LocalHFEmbedding(EMBED_MODEL_NAME)
    )


def init_neo4j_vector(embeddings_model: HuggingFaceEmbeddings, docs: List[Document]):
    """
    Try loading an existing Neo4j vector index; if absent, create a new one by indexing docs.
    """
    try:
        
        vectorstore = Neo4jVector.from_existing_index(
            embedding=embeddings_model,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PW,
            index_name=NEO4J_INDEX
        )
        print("loaded existing Neo4j vectorstore")
        return vectorstore
    except Exception as e:
        print(f"⚠️ Neo4j index '{NEO4J_INDEX}' not found; creating new vector store. ({e})")
        vectorstore = Neo4jVector.from_documents(
            documents=docs,
            embedding=embeddings_model,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PW,
            index_name=NEO4J_INDEX
        )
        print(f"✅ Created new Neo4j vector index '{NEO4J_INDEX}' with {len(docs)} docs.")
        return vectorstore

# --- Upsert to Chroma ---

def upsert_chroma(collection, docs: List[Document]):
    ids = [f"doc_{i}_{hash(d.page_content)}" for i, d in enumerate(docs)]
    collection.upsert(ids=ids,
                      documents=[d.page_content for d in docs],
                      metadatas=[d.metadata for d in docs])

# --- Retrieval methods ---

def bm25_retriever(docs: List[Document], query: str, k: int = 5) -> List[Document]:
    tokenized = [d.page_content.split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())
    chunks = [docs[idx] for idx, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]]
    print("BM25 chunks --> ", chunks)
    return chunks


def chroma_retriever(collection, query: str, k: int = 5) -> List[Document]:
    resp = collection.query(query_texts=[query], n_results=k,
                             include=["documents", "metadatas"])
    chunks =  [Document(page_content=d, metadata=m)
            for d, m in zip(resp["documents"][0], resp["metadatas"][0])]
    print("ChromaDB chunks --> ", chunks)
    return chunks


def neo4j_retriever(vectorstore: Neo4jVector, query: str, k: int = 5) -> List[Document]:
    chunks = vectorstore.similarity_search(query, k=k)
    print("Neo4j chunks --> ", chunks)
    return chunks

# --- Combine Common Context ---

def common_context(*lists_of_docs: List[Document]) -> str:
    sets = [set(d.page_content for d in docs) for docs in lists_of_docs]
    common = set.intersection(*sets)
    common_chunks = "\n\n".join(common)
    print( common_chunks)
    return common_chunks

# --- Ask Groq with combined context ---

def ask_groq(chroma_col, neo4j_vec, all_docs, question: str, k: int = 5):
    bm25_hits = bm25_retriever(all_docs, question, k)
    chroma_hits = chroma_retriever(chroma_col, question, k)
    neo4j_hits = neo4j_retriever(neo4j_vec, question, k)

    context = common_context(bm25_hits, chroma_hits, neo4j_hits)
    if not context:
        print("No common context across all retrievers.")
        return

    messages = [
        {"role": "system", "content": "You are an expert assistant. Answer using only provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(model=os.getenv("GROQ_MODEL_NAME", GROQ_MODEL_NAME),
                                         messages=messages, temperature=0.2)
    print(resp.choices[0].message.content)

# --- Main Flow ---

def main():
    # 1) Load or extract docs
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    docs = load_docs()
    if not docs:
        for name in PDF_FILES:
            path = os.path.join(PDF_DIR, f"{name}.pdf")
            if os.path.exists(path):
                docs.extend(extract_chunks_from_pdf(path, HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)))
        save_docs(docs)

    # 2) Init stores
    chroma_col = init_chroma()
    neo4j_vec = init_neo4j_vector(embeddings_model, docs)

    # 3) Upsert to Chroma if empty
    if chroma_col.count() == 0:
        upsert_chroma(chroma_col, docs)

    # 4) Interactive loop
    print("--- RAG System Ready ---")
    while True:
        q = input('❓ What do you want to know: ')
        if q.lower() == 'quit': break
        ask_groq(chroma_col, neo4j_vec, docs, q)

if __name__ == "__main__":
    main()
