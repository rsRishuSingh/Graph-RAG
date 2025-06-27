import os
import json
import time
import re
import fitz # PyMuPDF
from dotenv import load_dotenv
from typing import List
from io import BytesIO # Added for Chainlit PDF handling

# Langchain and LLM specific imports
from langchain.docstore.document import Document
from langchain_neo4j import Neo4jVector, Neo4jGraph # Kept imports for Neo4j functions, though not directly used in Chainlit flow
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_experimental.graph_transformers import LLMGraphTransformer # Kept imports for Neo4j functions
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

# ChromaDB specific imports
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction

# BM25 for keyword retrieval
from rank_bm25 import BM25Okapi

# Groq client for LLM inference
from groq import Groq

# Chainlit imports
import chainlit as cl
from engineio.payload import Payload

# Set max_decode_packets for Chainlit to handle larger messages (e.g., file uploads)
Payload.max_decode_packets = 500


# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Embedding model for vector similarity and semantic chunking (initialized globally for easy access)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# Groq LLM model name (defaults to "qwen/qwen3-32b" if not set in environment)
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "qwen/qwen3-32b")

# ChromaDB collection name and persistence path
COLLECTION_NAME = "TESLA_RAG_DOCS"
CHROMA_DB_PATH = "chromaDB/saved/"

# Path to store/load pre-chunked documents (used for initial loading or saving outside Chainlit's runtime)
ALL_DOCS_JSON = "all_docs.json"

# --- Custom EmbeddingFunction for ChromaDB ---
# This class wraps a SentenceTransformer model to be compatible with ChromaDB's
# embedding function interface, allowing local embedding generation.
class LocalHFEmbedding(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Encodes a batch of texts into embedding vectors."""
        return self.model.encode(texts).tolist()


# --- Text Chunking Functions ---

def recursive_split(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Splits text into smaller, overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
    This helps in handling large texts and preserving context across chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "] # Define preferred splitting points
    )
    return splitter.split_text(text)

def semantic_chunker(text: str) -> List[str]:
    """
    Splits text into semantically coherent chunks using LangChain's SemanticChunker.
    It first applies recursive splitting to break down very large inputs, then
    semantically chunks those segments using the global `embeddings_model`.
    """
    recursive_chunks = recursive_split(text)
    chunker = SemanticChunker(embeddings_model) # Uses the globally defined embeddings_model
    final_chunks: List[str] = []
    for chunk in recursive_chunks:
        semantic_chunks = chunker.split_text(chunk)
        final_chunks.extend(semantic_chunks)
    return final_chunks

# --- PDF Extraction and Document Management ---

def extract_chunks_from_pdf_bytes(pdf_bytes: bytes, name: str) -> List[Document]:
    """
    Reads a PDF file from its byte content, extracts text, applies semantic chunking,
    and returns a list of Document objects with relevant metadata (page, chunk, source).
    This function is designed for Chainlit's file upload mechanism.
    """
    docs: List[Document] = []
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_index, page in enumerate(pdf):
            text = page.get_text("text")
            # Clean up extra whitespace that often results from PDF text extraction
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                continue # Skip empty pages

            chunks = semantic_chunker(text) # Uses the harmonized semantic_chunker
            for chunk_index, chunk in enumerate(chunks):
                metadata = {
                    "page": page_index + 1,
                    "chunk": chunk_index,
                    "source": name # Use the provided name (e.g., filename) as source
                }
                docs.append(Document(page_content=chunk, metadata=metadata))
        pdf.close()
    except Exception as e:
        print(f"Error processing PDF from bytes ({name}): {e}")
    return docs

def load_docs(filepath: str = ALL_DOCS_JSON) -> List[Document]:
    """
    Loads a list of Document objects from a JSON file. This is useful for
    reusing pre-processed documents without re-extracting from PDFs every time.
    """
    print('⌛ Loading Chunks...')
    if not os.path.exists(filepath):
        print(f"No existing chunks found at {filepath}.")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        arr = json.load(f)
    print(f"Loaded {len(arr)} chunks from {filepath}.")
    return [Document(page_content=a["page_content"], metadata=a["metadata"]) for a in arr]

def save_docs(docs: List[Document], filepath: str = ALL_DOCS_JSON) -> None:
    """
    Saves a list of Document objects to a JSON file.
    """
    print('📥 Saving Chunks...')
    arr = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(docs)} documents to {filepath}")


# --- ChromaDB Operations ---

def create_or_reload_chroma_collection():
    """
    Initializes a PersistentClient for ChromaDB and either creates a new collection
    or loads an existing one with the specified embedding function.
    """
    print('🧩 Initializing ChromaDB client and collection...')
    client = PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=LocalHFEmbedding(EMBED_MODEL_NAME) # Ensures embeddings are consistent
    )
    print(f"ChromaDB collection '{COLLECTION_NAME}' is ready. Document count: {collection.count()}")
    return collection

def upsert_documents_to_chroma(docs: List[Document], collection) -> None:
    """
    Upserts (inserts or updates) a batch of Document objects into the given ChromaDB collection.
    Generates unique IDs based on document content for idempotency.
    """
    if not docs:
        print("No documents to upsert to ChromaDB.")
        return

    print(f'🧠 Storing {len(docs)} Embeddings in ChromaDB...')
    # Generate unique IDs for each document based on content hash to prevent duplicates
    ids = [f"doc_{i}_{hash(d.page_content)}" for i, d in enumerate(docs)]
    documents_content = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    try:
        collection.upsert(
            ids=ids,
            documents=documents_content,
            metadatas=metadatas
        )
        print("✅ Embeddings stored successfully in ChromaDB.")
    except Exception as e:
        print(f"Error upserting documents to ChromaDB: {e}")

def search_chroma(collection, query: str, k: int = 5) -> List[Document]:
    """
    Performs a pure vector-based k-Nearest Neighbors (k-NN) search in ChromaDB
    to find the most semantically similar documents to the query.
    """
    print(f"🔍 Performing vector search in ChromaDB for: '{query}'")
    resp = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"] # Include distances for potential advanced scoring/debugging
    )
    results = []
    if resp["documents"] and resp["documents"][0]:
        for doc_str, meta in zip(resp["documents"][0], resp["metadatas"][0]):
            results.append(Document(page_content=doc_str, metadata=meta))
    print(f"Found {len(results)} results from ChromaDB.")
    return results

# --- BM25 Retriever ---

def bm25_retriever(docs: List[Document], query: str, k: int = 5) -> List[Document]:
    """
    Performs BM25 keyword-based retrieval over the raw text content of Document list.
    BM25 is effective for exact keyword matches and term frequency-inverse document frequency.
    """
    if not docs:
        print("No documents available for BM25 retrieval.")
        return []

    print(f"🔎 Performing BM25 search for: '{query}'")
    texts = [d.page_content for d in docs]
    tokenized = [text.split() for text in texts] # Simple space tokenization for BM25
    bm25 = BM25Okapi(tokenized)
    
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    
    # Sort by score in descending order and get the top k document indices
    scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    top_idxs = [idx for idx, _ in scored_indices]

    results = [docs[i] for i in top_idxs]
    print(f"Found {len(results)} results from BM25.")
    return results

# --- Ensemble Retrieval ---

def ensemble_retrieval(
    all_docs: List[Document], # All original documents are needed for BM25 to search over
    chroma_collection,       # ChromaDB collection for vector search
    query: str,              # User's query string
    k: int = 5               # Number of top documents to retrieve after merging
) -> List[Document]:
    """
    Combines results from BM25 (keyword-based) and vector similarity (ChromaDB) retrieval.
    It gets top-k hits from both, assigns a score based on rank, merges, and de-duplicates
    to provide a richer set of context to the LLM.
    """
    print('🧐 Performing ensemble retrieval...')
    bm25_hits = bm25_retriever(all_docs, query, k)
    vec_hits = search_chroma(chroma_collection, query, k)

    scores = {} # Dictionary to store combined scores for each unique document content
    
    # Assign scores based on rank for BM25 hits (higher rank = higher score)
    for rank, doc in enumerate(bm25_hits):
        # Use page_content as the key to handle potential duplicates of content
        scores[doc.page_content] = scores.get(doc.page_content, 0) + (k - rank)
    
    # Assign scores based on rank for Vector hits, adding to existing scores if content overlaps
    for rank, doc in enumerate(vec_hits):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + (k - rank)

    # Sort the unique document contents by their combined score in descending order
    sorted_content_by_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    # Create a mapping from page_content to original Document object for quick reconstruction
    content_to_doc_map = {d.page_content: d for d in all_docs}

    # Reconstruct Document objects based on the sorted unique content
    retrieved_docs = []
    for content, _ in sorted_content_by_score:
        if content in content_to_doc_map: # Ensure the content exists in original docs
            retrieved_docs.append(content_to_doc_map[content])
    
    print(f"Ensemble retrieval found {len(retrieved_docs)} unique documents.")
    return retrieved_docs


# --- Neo4j Graph Operations ---
# These functions are included for completeness of the original combined system
# but are not directly called within the Chainlit application's main flow for simplicity.
# If you want to integrate graph-based retrieval, it would require additional Chainlit components.

def populate_neo4j_graph(docs: List[Document], neo4j_uri: str, neo4j_username: str, neo4j_password: str):
    """
    Initializes LLMGraphTransformer and populates the Neo4j graph database
    with entities and relationships extracted from the provided documents.
    """
    if not docs:
        print("No documents to populate Neo4j graph with.")
        return

    print("Connecting to Neo4j Aura for graph population...")
    try:
        llm_graph_transformer = ChatGroq(
            model_name=os.getenv("GROQ_MODEL_NAME", GROQ_MODEL_NAME),
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY")
        )
        transformer = LLMGraphTransformer(llm=llm_graph_transformer)

        graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo44j_username,
            password=neo4j_password,
        )
        print("✅ Connected to Neo4j Aura for graph population.")
        
        graph.refresh_schema() 
        print("Neo4j schema refreshed.")

        print(f"Starting graph transformation and insertion for {len(docs)} documents...")
        start_time = time.time()
        for idx, doc in enumerate(docs):
            print(f"[{idx+1}/{len(docs)}] Converting → GraphDocument and adding to Neo4j...", end="\r")
            try:
                gd = transformer.convert_to_graph_documents([doc])[0]
                graph.add_graph_documents(
                    graph_documents=[gd],
                    include_source=True,
                    baseEntityLabel=True,
                )
            except Exception as e:
                print(f"\n⚠️ Error on chunk {idx+1}: {e!r} — retrying this chunk...")
                try:
                    gd = transformer.convert_to_graph_documents([doc])[0]
                    graph.add_graph_documents(
                        graph_documents=[gd],
                        include_source=True,
                        baseEntityLabel=True,
                    )
                except Exception as retry_e:
                    print(f"❌ Failed after retry on chunk {idx+1}: {retry_e!r} - skipping this chunk.")
        print(f"\n🏁 Done with Neo4j graph population! Total time: {time.time() - start_time:.1f}s")
    except Exception as e:
        print(f"Error connecting to or populating Neo4j: {e}")

# --- Results Printer (not actively used in Chainlit's runtime output, but useful for debugging) ---

def print_results(results: List[Document]) -> None:
    """Uniformly prints out snippets and metadata of retrieved Documents."""
    if not results:
        print("No results to display.")
        return
    for i, doc in enumerate(results, 1):
        snippet = re.sub(r'\s+', ' ', doc.page_content).strip()[:400]
        print(f"--- Result {i} ---")
        print(f"Snippet : {snippet}{'...' if len(doc.page_content) > 400 else ''}")
        print(f"Metadata: {doc.metadata}\n")

# --- Ask Groq ---

def ask_Groq(
    collection, # The ChromaDB collection for vector search
    k: int,      # Number of top documents to retrieve for context
    docs: List[Document], # The complete list of documents for BM25 retrieval
    question: str # The user's question
) -> str:
    """
    Retrieves relevant context using ensemble retrieval (BM25 + Vector Search)
    and then queries the Groq LLM to generate an answer based on that context.
    """
    print(f"\n💬 Asking Groq for: '{question}'")
    # Perform ensemble retrieval to get the most relevant document chunks
    retrieved_docs = ensemble_retrieval(docs, collection, question, k)
    
    if not retrieved_docs:
        print("No relevant context found to answer the question.")
        return "🤖 I couldn't find enough information to answer that question from the documents."

    # Combine the page content of the retrieved documents into a single context string
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Prepare messages for the Groq API call
    system_msg = {"role": "system", "content": "You are an expert assistant. Answer the user's question concisely and accurately based ONLY on the provided context. If the answer is not in the context, state that you don't have enough information."}
    user_msg = {
        "role": "user",
        "content": (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
    }

    # Retrieve Groq API key from environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment variables.")
    client = Groq(api_key=groq_api_key)

    try:
        # Make the API call to Groq
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL_NAME", GROQ_MODEL_NAME), # Use model from env or default
            messages=[system_msg, user_msg],
            temperature=0.2, # Low temperature for more factual, less creative answers
            stream=False,    # Non-streaming response
        )
        answer = resp.choices[0].message.content

        # Remove internal "think" blocks that LLMs might generate for their thought process
        cleaned_answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
        return cleaned_answer.strip()

    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "🤖 I encountered an error while trying to generate a response."


# --- Chainlit Event Handlers (Replaces the traditional `if __name__ == "__main__": main()` block) ---

@cl.on_chat_start
async def setup():
    """
    This function is automatically called by Chainlit when a new chat session starts.
    It handles PDF file uploads, processes them, and sets up the knowledge base (ChromaDB).
    """
    # Prompt the user to upload a PDF file
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF to build the knowledge base. Max size: 20MB.",
            accept=["application/pdf"],
            max_size_mb=20, # Set a maximum file size
            timeout=180 # Set a timeout for the user to upload the file
        ).send()
    
    # Get the first uploaded PDF file object
    pdf_file = files[0]
    
    # Send an initial message to the user indicating processing has started
    msg = cl.Message(content=f"Processing `{pdf_file.name}`...")
    await msg.send()

    try:
        # Read the PDF content as bytes from the uploaded file
        pdf_bytes = BytesIO(pdf_file.content).getvalue()
        
        # Extract document chunks from the PDF bytes
        docs = extract_chunks_from_pdf_bytes(pdf_bytes, pdf_file.name)
        
        # Update the processing message with the number of chunks extracted
        msg.content = f"Extracted {len(docs)} chunks from `{pdf_file.name}`."
        await msg.update()

        if not docs:
            # If no content could be extracted, inform the user and exit setup
            await cl.Message(content="No content could be extracted from the PDF. Please try a different file.").send()
            return

        # Initialize or reload the ChromaDB collection
        collection = create_or_reload_chroma_collection()

        # Upsert documents into ChromaDB only if the collection is empty
        # This prevents re-uploading the same documents if the app is restarted
        # without deleting the persistent ChromaDB storage.
        if collection.count() == 0:
            upsert_documents_to_chroma(docs, collection)
            msg.content = f"Extracted {len(docs)} chunks from `{pdf_file.name}` and stored embeddings in ChromaDB."
            await msg.update()
        else:
            # If ChromaDB already has documents, inform the user.
            # For this example, we proceed with the existing collection.
            msg.content = f"ChromaDB already contains documents. Loaded existing knowledge base. If you upload a new PDF, please be aware the existing data remains."
            await msg.update()
            # In a more advanced app, you might ask the user if they want to
            # clear the old data or append to it.

        # Store the ChromaDB collection object and the list of all documents
        # in the Chainlit user session, making them accessible in subsequent messages.
        cl.user_session.set("collection", collection)
        cl.user_session.set("docs", docs) # Store all docs for BM25 retrieval

        # Send a final message to indicate the knowledge base is ready
        await cl.Message(content="PDF loaded and knowledge base is ready! Ask me anything about the document.").send()

    except Exception as e:
        # Catch any errors during setup and inform the user
        await cl.Message(content=f"An error occurred during setup: {e}. Please try again.").send()
        print(f"Error during setup: {e}")

@cl.on_message
async def chat(message: cl.Message): # Chainlit passes the user's message as a cl.Message object
    """
    This function is called by Chainlit every time the user sends a message.
    It retrieves the knowledge base from the session, performs RAG (Retrieval Augmented Generation),
    and sends the LLM's answer back to the user.
    """
    # Retrieve the ChromaDB collection and all documents from the user session
    collection = cl.user_session.get("collection")
    docs = cl.user_session.get("docs")

    # Check if the knowledge base is properly set up
    if not collection or not docs:
        await cl.Message(content="Knowledge base not set up. Please upload a PDF first by restarting the chat (`/reset` command).").send()
        return

    # Send a processing message to the user
    processing_msg = cl.Message(content=f"Searching for relevant information for: '{message.content}'...")
    await processing_msg.send()

    # Call the ask_Groq function to get the answer from the LLM
    # We retrieve the top 5 relevant documents (k=5) for context
    answer = ask_Groq(collection, 5, docs, message.content)
    
    # Update the processing message with the final answer
    processing_msg.content = answer
    await processing_msg.update()

# The original `main` function is commented out as its functionality is now handled by Chainlit decorators.
# if __name__ == "__main__":
#     main()
