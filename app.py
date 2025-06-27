import os
import json
import re
from dotenv import load_dotenv
from typing import List

import chainlit as cl
import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_neo4j import Neo4jVector
from groq import Groq

load_dotenv()

# Configuration
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "qwen/qwen3-32b")
COLLECTION_NAME = "TESLA_RAG_DOCS"
CHROMA_DB_PATH = "chromaDB/saved/"
ALL_DOCS_JSON = "all_docs.json"
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PW = os.getenv("NEO4J_PASSWORD")
NEO4J_INDEX = os.getenv("NEO4J_INDEX", "LangChainDocs")

# ------- Splitting & Chunking -------
def recursive_split(text: str, chunk_size=500, chunk_overlap=100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text)


def semantic_chunker(text: str, embed_model) -> List[str]:
    out = []
    for seg in recursive_split(text):
        chunker = SemanticChunker(embed_model)
        out.extend(chunker.split_text(seg))
    return out

# ------- Load/Save Documents -------
def save_docs(docs: List[Document], filepath: str = ALL_DOCS_JSON) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            f,
            indent=2,
        )


def load_docs(filepath: str = ALL_DOCS_JSON) -> List[Document]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=i["page_content"], metadata=i["metadata"]) for i in data]

# ------- ChromaDB -------
def get_chroma_collection() -> Chroma:
    embed_fn = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

# ------- Neo4j Vector -------
def get_neo4j_vector(docs: List[Document]) -> Neo4jVector:
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    try:
        return Neo4jVector.from_existing_index(
            embedding=embed,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PW,
            index_name=NEO4J_INDEX,
        )
    except:
        return Neo4jVector.from_documents(
            documents=docs,
            embedding=embed,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PW,
            index_name=NEO4J_INDEX,
        )

# ------- RAG QA -------
def ask_Groq(collection: Chroma, docs: List[Document], k: int, query: str) -> str:
    bm25 = BM25Retriever.from_texts(
        [d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
        k=k,
    )
    chroma_ret = collection.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    neo4j_ret = get_neo4j_vector(docs).as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    ens = EnsembleRetriever(
        retrievers=[bm25, chroma_ret, neo4j_ret], weights=[1, 1, 1]
    )
    top = ens.invoke(query)
    context = "\n\n".join([d.page_content for d in top])
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    messages = [
        {"role": "system", "content": "You are an expert assistant. Use only provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
    ]
    resp = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content

# ------- Chainlit -------
@cl.on_chat_start
async def setup():
    # load existing docs
    docs = load_docs()
    if not docs:
        await cl.Message("⚠️ No saved documents found. Please run the pipeline offline first.").send()
        return
    # load or create collection
    collection = get_chroma_collection()
    if not collection._collection.count():
        collection.add_documents(
            [d.page_content for d in docs], metadatas=[d.metadata for d in docs]
        )
        collection.persist()
    # save session
    cl.user_session.set("collection", collection)
    await cl.Message("✅ Collection loaded. Ask me anything!").send()

@cl.on_message
async def chat(message: str):
    collection = cl.user_session.get("collection")
    docs = load_docs("all_docs.json")
    answer = ask_Groq(collection, docs, k=5, query=message.content)
    await cl.Message(answer).send()
