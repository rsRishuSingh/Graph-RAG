
import os
import json
import fitz  # PyMuPDF
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from chromadb import PersistentClient
from chromadb.config import Settings
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from chromadb.utils.embedding_functions import EmbeddingFunction

#  Custom EmbeddingFunction for ChromaDB using local SentenceTransformer model
class LocalHFEmbedding(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into embedding vectors."""
        return self.model.encode(texts).tolist()

#  Configuration
COLLECTION_NAME   = "ch_DB_TRUMP"
PDF_DIR           = "PDFs/"
PDF_FILES         = ["Trump"]           # without .pdf extension
EMBED_MODEL_NAME  = "Qwen/Qwen3-Embedding-0.6B"
embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME) # required in only semantic chunking

#  Semantic chunker wrapper
def semantic_chunker(text: str) -> List[str]:
    """
    Split text into semantically coherent chunks using LangChain's SemanticChunker.
    """
    chunker = SemanticChunker(embeddings_model)
    return chunker.split_text(text)

#  PDF Extraction and Chunking
def extract_chunks_from_pdf(pdf_path: str) -> List[Document]:
    """
    Reads a PDF file, splits each page into semantic chunks, and
    returns a list of Document objects with page/chunk metadata.
    """
    docs: List[Document] = []
    pdf = fitz.open(pdf_path)
    for page_index, page in enumerate(pdf):
        text = page.get_text("text")
        chunks = semantic_chunker(text)
        for chunk_index, chunk in enumerate(chunks):
            metadata = {
                "page": page_index + 1,
                "chunk": chunk_index,    # set back to zero when page changes
                "source": os.path.basename(pdf_path)
            }
            docs.append(Document(page_content=chunk, metadata=metadata))
    pdf.close()
    return docs

#  ChromaDB Client Initialization
def create_or_reload_collection():
    """
    Creates (or loads, if exists) a ChromaDB collection with a local HF embedding.
    """
    client = PersistentClient(path="chromaDB/saved/")

    collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=LocalHFEmbedding(EMBED_MODEL_NAME)
)
    return collection

#  Document Upsert
def upsert_documents(
    docs: List[Document],
    collection
) -> None:
    """
    Upserts a batch of Document objects into the given ChromaDB collection.
    """
    ids = [f"id_{i}" for i in range(len(docs))]
    documents = [d.page_content for d in docs]
    metadatas = [d.metadata    for d in docs]
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

#  Delete Entire Collection
def delete_collection(collection) -> None:
    """
    Deletes all entries from the given ChromaDB collection.
    """
    collection.delete()

#  Vector Similarity Search (Chroma)
def search_chroma(collection, query: str, k: int = 5) -> List[Document]:
    """
    Performs a pure vector-based k-NN search in ChromaDB.
    """
    resp = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"]
    )
    # Reconstruct Documents from response
    results = []
    for doc_str, meta in zip(resp["documents"][0], resp["metadatas"][0]):
        results.append(Document(page_content=doc_str, metadata=meta))
    return results

#  BM25 Retriever
def bm25_retriever(
    docs: List[Document],
    query: str,
    k: int = 5
) -> List[Document]:
    """
    Performs BM25 retrieval over the raw chunk texts of Document list.
    """
    texts = [d.page_content for d in docs]
    tokenized = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in top_idxs]

#  Ensemble Retrieval: BM25 + Vector
def ensemble_retrieval(
    docs: List[Document],
    collection,
    query: str,
    k: int = 5
) -> List[Document]:
    """
    Combines BM25 and vector retrieval:
    - gets top-k BM25 hits and top-k Chroma hits,
    - scores them by rank, merges & de-duplicates.
    """
    bm25_hits = bm25_retriever(docs, query, k)
    vec_hits  = search_chroma(collection, query, k)

    scores = {}
    for rank, doc in enumerate(bm25_hits):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + (k - rank)
    for rank, doc in enumerate(vec_hits):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + (k - rank)

    sorted_texts = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
  
    # Map back to Document objects
    lookup = {d.page_content: d for d in docs}
    return [lookup[text] for text, _ in sorted_texts]

#  Results Printer
def print_results(results: List[Document]) -> None:
    """
    Uniformly prints out snippets and metadata of retrieved Documents.
    """
    for i, doc in enumerate(results, 1):
        snippet = doc.page_content.replace("\n", " ")[:200]
        print(f"--- Result {i} ---")
        print(f"Snippet : {snippet}...")
        print(f"Metadata: {doc.metadata}\n")

#  JSON Save & Load for Documents
def save_docs(docs: List[Document], filepath: str = "all_docs.json") -> None:
    """
    Saves a list of Document objects to JSON for reuse.
    """
    arr = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(docs)} documents to {filepath}")

def load_docs(filepath: str = "all_docs.json") -> List[Document]:
    """
    Loads Document objects from a JSON file.
    """
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        arr = json.load(f)
    return [Document(page_content=a["page_content"], metadata=a["metadata"]) for a in arr]

#  Main Execution Flow
if __name__ == "__main__":
    
    # 1) Load or build document chunks
    docs = load_docs()  # remove this when deployed because it prevent newer docs from getting stored
    if not docs:
        for name in PDF_FILES:
            path = os.path.join(PDF_DIR, f"{name}.pdf")
            docs.extend(extract_chunks_from_pdf(path))
        save_docs(docs)

    # 2) Initialize or load ChromaDB collection
    collection = create_or_reload_collection()

    # 3) Upsert docs into Chroma (if first run)
    if not collection.count():  # only upsert if empty
        upsert_documents(docs, collection)

    # 4) Example queries
    query = "where trump born"

    # print("\n[Vector Search]")
    # print_results(search_chroma(collection, query, k=5))

    # print("\n[BM25 Search]")
    # print_results(bm25_retriever(docs, query, k=5))

    # print("\n[Ensemble Search]")
    # print_results(ensemble_retrieval(docs, collection, query, k=5))
