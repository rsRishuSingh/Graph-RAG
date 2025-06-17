import re
import os
import json
import fitz  # PyMuPDF
from groq import Groq
from typing import List
from rank_bm25 import BM25Okapi
from chromadb import PersistentClient
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
#  Custom EmbeddingFunction for ChromaDB using local SentenceTransformer model
class LocalHFEmbedding(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into embedding vectors."""
        return self.model.encode(texts).tolist()

#  Configuration
EMBED_MODEL_NAME  = "Qwen/Qwen3-Embedding-0.6B"
GROQ_MODEL_NAME = "qwen/qwen3-32b"
COLLECTION_NAME   = "ch_DB_TRUMP"
PDF_DIR           = "PDFs/"
PDF_FILES         = ["Trump"]           # without .pdf extension

# login via terminal and set huggingface api key
embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME) # required in only semantic chunking


def recursive_split(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
        )
    return splitter.split_text(text)

#  Semantic chunker wrapper
def semantic_chunker(text: str) -> List[str]:
    """
    Split text into semantically coherent chunks using LangChain's SemanticChunker.
    """
    recursiveChunks = recursive_split(text);
    chunker = SemanticChunker(embeddings_model)
    final_chunks = []
    for chunk in recursiveChunks:
        semantic_chunks = chunker.split_text(chunk)
        final_chunks.extend(semantic_chunks)   # list + list 
    return final_chunks

#  PDF Extraction and Chunking
def extract_chunks_from_pdf(pdf_path: str) -> List[Document]:
    """
    Reads a PDF file, splits each page into semantic chunks, and
    returns a list of Document objects with page/chunk metadata.
    """
    print('🗂️  Getting PDF...\n\n')
    docs: List[Document] = []
    pdf = fitz.open(pdf_path)
    for page_index, page in enumerate(pdf):
        print('📖 Reading Page no: ', page_index+1)
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
    print('🧩 Creating Database...')
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
    print('🧠 Storing Embeddings...')
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
    print('📥 Saving Chunks...')
    arr = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(docs)} documents to {filepath}")

# Load already saved json
def load_docs(filepath: str = "all_docs.json") -> List[Document]:
    """
    Loads Document objects from a JSON file.
    """
    print('⌛ Loading Chunks...')
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        arr = json.load(f)
    return [Document(page_content=a["page_content"], metadata=a["metadata"]) for a in arr]

# Ask Groq
def ask_Groq(collection, docs, k, question: str = "generate a summary of this pdf?"):
    # Retrieve top‑k docs
    docs = ensemble_retrieval(docs, collection, question, k)
    context = "\n\n".join(doc.page_content for doc in docs)

    #Prepare your messages
    system_msg = {"role": "system",    "content": "You are an expert assistant."}
    user_msg   = {
        "role": "user",
        "content": (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
    }

    # (only the API key here)
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    resp = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL_NAME", "qwen/qwen3-32b"),
        messages=[system_msg, user_msg],
        temperature=0.2,
        stream=False,
    )
    answer = resp.choices[0].message.content
    think_match = re.search(r"<think>(.*?)</think>", answer, flags=re.DOTALL)
    if think_match:
        think_text = think_match.group(1).strip()
        print("💬\n", think_text)

    # Remove the <think>...</think> block from the original answer
    cleaned = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
    print("\n\n🤖", cleaned.strip())
        


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
    
    
    # 4) Ask groq
    while(True):
        question = input('❓ What do you want to know: ')
        if(question.capitalize() == 'Quit'):
            break
        ask_Groq( collection, docs, 3, question)
    print('👋 Bye! See you Again')
    
    # print("\n[Vector Search]")
    # print_results(search_chroma(collection, query, k=5))

    # print("\n[BM25 Search]")
    # print_results(bm25_retriever(docs, query, k=5))

    # print("\n[Ensemble Search]")
    # print_results(ensemble_retrieval(docs, collection, query, k=5))
