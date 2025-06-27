import os
import json
import getpass
from dotenv import load_dotenv
from typing import List

from langchain.docstore.document import Document
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def load_docs(filepath: str) -> List[Document]:
    """Load your pre-chunked JSON into Document objects."""
    with open(filepath, "r", encoding="utf-8") as f:
        arr = json.load(f)
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in arr]


def get_groq_api_key() -> str:
    """Retrieve GROQ_API_KEY from env or prompt the user."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        key = getpass.getpass("Enter your Groq API key: ")
    return key


def main():
    # Load environment variables
    load_dotenv()

    # 1) Load and embed documents
    docs = load_docs("all_docs.json")
    print(f"🔖 Loaded {len(docs)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2) Build or reuse the Neo4j vector store
    url      = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]

    vectorstore = Neo4jVector.from_documents(
        documents=docs,
        embedding=embeddings,
        url=url,
        username=username,
        password=password,
        index_name="LangChainDocs",
    )
    print("✅ Vector store is ready in Neo4j")

    # 3) Initialize the LLM
    groq_key = get_groq_api_key()
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=groq_key
    )

    # 4) Define a custom prompt for financial QA
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a financial assistant. Answer from the given context and do not hallucinate.\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
    )

    # 5) Set up RetrievalQA chain with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # 6) Run a sample query using invoke() to avoid deprecation warning
    user_question = input("Enter your financial question: ")
    result = qa_chain.invoke({"query": user_question})

    # 7) Display results
    print("\n🧠 Answer:", result["result"])
    print("\n📄 Source Documents:")
    for doc in result["source_documents"]:
        score = doc.metadata.get('score', 'N/A')
        source = doc.metadata.get('source', 'unknown')
        print(f"- {source} (score: {score})")


if __name__ == "__main__":
    main()
