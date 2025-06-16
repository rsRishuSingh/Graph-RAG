# docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant
import os
import fitz  # PyMuPDF
import json
from typing import List, Dict
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from rank_bm25 import BM25Okapi

# CONFIG
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "q_DB_For_TESLA"
PDF_DIR = "PDFs/"
PDF_FILES = ["TESLANEW"]
THRESHOLD = 0.75

# EMBEDDING MODEL
embeddings_model = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

# SEMANTIC CHUNKER
def semantic_chunker(text):
    """Split text into semantically coherent chunks."""
   
    chunker = SemanticChunker(embeddings_model)
    print(chunker)
    return chunker.split_text(text)

# PDF EXTRACTION AND CHUNKING
def extract_chunks_from_pdf(pdf_path):
    """Extracts semantic chunks from a PDF and returns as Documents with metadata."""
 
    docs = []
    print(pdf_path)
    pdf = fitz.open(pdf_path)
    for page_index, page in enumerate(pdf):
        
        page_no = page_index + 1
        print(page)
        text = page.get_text("text")
        chunks = semantic_chunker(text) # (ideal chunk size = 500 tokens ) 
        for chunk_index, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "page": page_no,
                    "chunk": chunk_index,
                    "source": os.path.basename(pdf_path)
                }
            ))
    pdf.close()
    return docs

# QDRANT OPERATIONS
def create_or_reload_qdb(all_docs):
    """Creates the Qdrant collection if missing or loads existing one."""
    
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
    collections = [c.name for c in client.get_collections().collections]
    
    if COLLECTION_NAME not in collections:
        
        print(f"Creating and seeding collection: {COLLECTION_NAME}")
        return Qdrant.from_documents(
            documents=all_docs,
            embedding=embeddings_model,
            url=QDRANT_URL,
            prefer_grpc=False,
            collection_name=COLLECTION_NAME
        )
    else:
        
        print(f"Using existing collection: {COLLECTION_NAME}")
        return Qdrant(
            client=client,
            embeddings=embeddings_model,
            collection_name=COLLECTION_NAME
        )


def delete_qdb():
    """Deletes the Qdrant collection entirely."""
    
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
    collections = [c.name for c in client.get_collections().collections]
    
    if COLLECTION_NAME in collections:
        print(f"Deleting collection: {COLLECTION_NAME}")
        client.delete_collection(collection_name=COLLECTION_NAME)
    else:
        print(f"Collection {COLLECTION_NAME} does not exist.")


# RETRIEVAL FUNCTIONS
def search_qdb(qdrant, query, k = 5):
    """Performs a pure vector-based similarity search (k-NN)."""
   
    return qdrant.similarity_search(query=query, k=k)


def cosine_search(qdrant, query, k = 5):
    """Alias for vector k-NN (cosine) retrieval."""
    
    # Under the hood, Qdrant uses cosine similarity by default
    return search_qdb(qdrant, query=query, k=k)


def bm25_retriever(all_docs, query, k = 5):
    """Performs BM25 retrieval over the raw chunk texts."""
    
    tokenized = [doc.page_content.split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())
    
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [all_docs[i] for i in top_idxs]


def ensemble_retrieval(all_docs, qdrant, query, k = 5):
    """
    Combines BM25 and cosine retrieval:
    - Get top-k from BM25 and top-k from vector search
    - Merge, dedupe, rank by combined score
    """
    bm25_hits = bm25_retriever(all_docs, query, k)
    vec_hits = cosine_search(qdrant, query, k)
    
    scores: Dict[str, float] = {}
    for rank, doc in enumerate(bm25_hits):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + (k - rank)
    
    for rank, doc in enumerate(vec_hits):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + (k - rank)
    
    sorted_texts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    combined = [next(d for d in all_docs if d.page_content == text) for text, _ in sorted_texts[:k]]
    return combined


# RESULT PRINTER
def print_results(results):
    """Prints a uniform list of retrieval results."""
    
    for i, doc in enumerate(results, 1):
        snippet = doc.page_content.replace("\n", " ")[:200]
        print(f"--- Result {i} ---")
        print(f"Snippet : {snippet}...")
        print(f"Metadata: {doc.metadata}\n")


def save_docs(docs, filepath = "all_docs.json"):
    """Save list of Documents to a JSON file."""

    arr = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(docs)} documents to {filepath}")


def load_docs(filepath = "all_docs.json"):
    """Load Documents list from a JSON file."""
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []
    
    with open(filepath, "r", encoding="utf-8") as f:
        arr = json.load(f)

    docs = [Document(page_content=a["page_content"], metadata=a["metadata"]) for a in arr]
    print(f"Loaded {len(docs)} documents from {filepath}")
    return docs


# MAIN
if __name__ == "__main__":
   
   # Load docs if available else create and store
    all_docs = []
    all_docs = load_docs()
    if not all_docs:
        
        for name in PDF_FILES:
            path = os.path.join(PDF_DIR, f"{name}.pdf")
            all_docs.extend(extract_chunks_from_pdf(path))
        save_docs(all_docs)
   
   
    # Create or load collection
    # qdb = create_or_reload_qdb(all_docs)

    # Example usage
    query = "Elon Musk"
    # print("\n[Vector Search]")
    # print_results(cosine_search(qdb, query, k=5))

    print("\n[BM25 Search]")
    print_results(bm25_retriever(all_docs, query, k=5))

    # print("\n[Ensemble Search]")
    # print_results(ensemble_retrieval(all_docs, qdb, query, k=5))
