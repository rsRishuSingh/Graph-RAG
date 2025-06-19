from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import EmbeddingFunction
from chromadb import PersistentClient

class LocalHFEmbedding(EmbeddingFunction):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input).tolist()



chroma_client = PersistentClient(path="chromaDB/saved/")
#All writes (add, delete, etc.) are auto-persisted to that folder — you don’t call persist() manually.


collection = chroma_client.get_or_create_collection(
    name="my_collection",
    embedding_function=LocalHFEmbedding("Qwen/Qwen3-Embedding-0.6B")
)

collection.add(
    documents=[
        "pineapple juice come with pineapple price from nearby farm",
        "He lives in USA works for American company their not from india"
    ],
    metadatas=[{'source':'my_source', 'page':1}, {'source':'my_source', 'page':1}],
    ids=["id1", "id2"]
)

query1 = "where does he lives"
query2 = "order juice"

results = collection.query(
    query_texts=[query1, query2], # Chroma will embed this for you and return list of list
    n_results=1 # how many results to return,
)
print(results)

'''
{
'ids': [['id1', 'id2']], 
'embeddings': None, 
'documents': [['This is a document about pineapple', 'This is a document about oranges']], 
'uris': None, 
'included': ['metadatas', 'documents', 'distances'], 
'data': None, 
'metadatas': [[{'source': 'my_source', 'page': 1}, {'source': 'my_source', 'page': 1}]],
'distances': [[1.0404009819030762, 1.2430799007415771]]
 }
'''

# results = collection.query(
#     query_texts=["This is a query document about hawaii"],
#     n_results=2, 
#     include=['embeddings']
# )
# print(results['embeddings'])

# chroma_client.persist(it requiers path here ?)

#Reload DB
# chroma_client = chromadb.Client(Settings(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory="./chroma_db"
# ))

# collection = chroma_client.get_collection("my_collection")
# print(collection.get())  # Returns stored data
