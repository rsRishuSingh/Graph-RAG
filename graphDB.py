import os
import json
import time
from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain_neo4j import Neo4jGraph
from typing import List

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

llm = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.7, api_key=GROQ_API_KEY)
transformer = LLMGraphTransformer(llm=llm)


def load_docs(filepath: str) -> List[Document]:
    print("⌛ Loading chunks from", filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        arr = json.load(f)
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in arr]

all_docs = load_docs("all_docs.json")
print(f"📄 Loaded {len(all_docs)} pre-chunked documents")

graph = Neo4jGraph(
    url="bolt+s://933bd554.databases.neo4j.io:7687",
    username="neo4j",
    password="CMy8sjFM9grBNaz0R4uPWvMznKb8ECad-buNINUPcbs",
    database="neo4j",
)

print("✅ Connected to Neo4j Aura")

start = time.time()
for idx, doc in enumerate(all_docs[788:], 788):
    print(f"[{idx}/{len(all_docs)}] Converting → GraphDocument…", end="\r")
    try:
        gd = transformer.convert_to_graph_documents([doc])[0]
    except Exception as e:
        print(f"\n⚠️ Error on chunk {idx}: {e!r} — retrying…")
        gd = transformer.convert_to_graph_documents([doc])[0]

    graph.add_graph_documents(
        graph_documents=[gd],
        include_source=True,
        baseEntityLabel=True,
    )

print(f"\n🏁 Done! Total time: {time.time() - start:.1f}s")
