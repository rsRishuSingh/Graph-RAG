import os
import re
import fitz  # PyMuPDF
import chainlit as cl
from io import BytesIO
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from groq import Groq

# Configuration constants
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "qwen/qwen3-32b")
COLLECTION_NAME = "ch_DB_TRUMP"
embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME) # required in only semantic chunking

# Custom EmbeddingFunction for ChromaDB using local SentenceTransformer model
class LocalHFEmbedding(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into embedding vectors."""
        return self.model.encode(texts).tolist()

# Recursive split using character boundaries

def recursive_split(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

# Semantic chunker wrapper

def semantic_chunker(text: str) -> List[str]:
    """
    Split text into semantically coherent chunks using SemanticChunker.
    """
    recursive_chunks = recursive_split(text)
    chunker = SemanticChunker(embeddings_model)
    final_chunks: List[str] = []
    for chunk in recursive_chunks:
        semantic_chunks = chunker.split_text(chunk)
        final_chunks.extend(semantic_chunks)
    return final_chunks

# Extract chunks from PDF

def extract_chunks_from_pdf_bytes(pdf_bytes: bytes, name: str) -> List[Document]:
    """
    Reads a PDF from bytes, applies semantic chunking, and returns Document objects.
    """
    docs: List[Document] = []
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_index, page in enumerate(pdf):
        text = page.get_text("text")
        chunks = semantic_chunker(text)
        for chunk_index, chunk in enumerate(chunks):
            metadata = {
                "page": page_index + 1,
                "chunk": chunk_index,
                "source": name
            }
            docs.append(Document(page_content=chunk, metadata=metadata))
    pdf.close()
    return docs

# Initialize or reload ChromaDB collection

def create_or_reload_collection():
    """
    Creates or loads a ChromaDB collection with local HF embeddings.
    """
    client = PersistentClient(path="chromaDB/saved/")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=LocalHFEmbedding(EMBED_MODEL_NAME)
    )
    return collection

# Upsert documents

def upsert_documents(docs: List[Document], collection) -> None:
    """
    Upserts batch of Document objects into ChromaDB.
    """
    ids = [f"id_{i}" for i in range(len(docs))]
    documents = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

# Vector search

def search_chroma(collection, query: str, k: int = 5) -> List[Document]:
    """
    k-NN search in ChromaDB.
    """
    resp = collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas"])
    results: List[Document] = []
    for doc_str, meta in zip(resp["documents"][0], resp["metadatas"][0]):
        results.append(Document(page_content=doc_str, metadata=meta))
    return results

# BM25 retriever

def bm25_retriever(docs: List[Document], query: str, k: int = 5) -> List[Document]:
    """
    BM25 retrieval over raw chunk texts.
    """
    texts = [d.page_content for d in docs]
    tokenized = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in top_idxs]

# Ensemble retrieval
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
    print('🧐 Searching in DB')
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

# Ask Groq model

def ask_Groq(collection, docs: List[Document], k: int, question: str) -> str:
    """
    Retrieve top-k docs and ask Groq for answer.
    """
    # hits = ensemble_retrieval(docs, collection, question, k)
    hits = search_chroma(collection, question, k)
    context = "\n\n".join(d.page_content for d in hits)
    system_msg = {"role": "system", "content": "You are an expert assistant."}
    user_msg = {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n"}
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(model=GROQ_MODEL_NAME, messages=[system_msg, user_msg], temperature=0.2)
    answer = resp.choices[0].message.content
    # Remove internal think blocks
    cleaned = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
    return cleaned.strip()

# Chainlit events

@cl.on_chat_start
async def setup():
    # Ask user to upload a PDF
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF to build the knowledge base.",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180
        ).send()
    pdf = files[0]
    with open(pdf.path, "rb") as f:
        pdf_bytes = f.read()
    # print(dir(cl.AskFileMessage)) 
    await cl.Message(content=f"Processing `{pdf.name}`...").send()

    # Extract and store docs
    docs = extract_chunks_from_pdf_bytes(pdf_bytes, pdf.name)
    # Persist to ChromaDB
    collection = create_or_reload_collection()
    if not collection.count():
        upsert_documents(docs, collection)

    # Save in session
    cl.user_session.set("collection", collection)
    cl.user_session.set("docs", docs)

    await cl.Message(content="PDF loaded! Ask me anything.").send()

@cl.on_message
async def chat(message: str):
    collection = cl.user_session.get("collection")
    docs = cl.user_session.get("docs")
    query = message.content
    answer = ask_Groq(collection, docs, 3, query)
    await cl.Message(content=answer).send()
