import os
import json
from dotenv import load_dotenv
from typing import List

from langchain.docstore.document import Document
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings


def load_docs(filepath: str) -> List[Document]:
    """Load your pre-chunked JSON into Document objects."""
    with open(filepath, "r", encoding="utf-8") as f:
        arr = json.load(f)
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in arr]

def main():
    load_dotenv()

    # 1) Load chunks
    docs = load_docs("all_docs.json")
    print(f"🔖 Loaded {len(docs)} chunks")

    # 2) Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3) Neo4j creds
    url      = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]

    # 4) Build/reuse the vector store
    vectorstore = Neo4jVector.from_documents(
        documents=docs,
        embedding=embeddings,
        url=url,
        username=username,
        password=password,
        index_name="LangChainDocs",  # reuse the same index
    )
    print("✅ Vector store is ready in Neo4j")

    # 5) Similarity search
    query = "revenue of tesla"
    results = vectorstore.similarity_search_with_score(query, k=5)

    print("\nTop 5 results:")
    for idx, (doc, score) in enumerate(results, start=1):
        print(f"\n[{idx}] score={score:.4f}\n{doc.page_content}")

if __name__ == "__main__":
    main()
