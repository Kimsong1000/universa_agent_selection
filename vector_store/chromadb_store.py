import chromadb
from chromadb.config import Settings
from chromadb.api.types import QueryResult, GetResult
from embedding_functions.base_embedder import BaseEmbeddingFunction
 
class ChromaDB:
    def __init__(self, db_path, embedding_function=None):
        self.client = chromadb.Client(Settings(persist_directory=db_path))
        self.embedding_function = embedding_function
        self.collection = self.client.get_or_create_collection(name="agents_collection")
 
    def add_data(self, documents, ids, metadatas=None):
        params = {
            "documents": documents,
            "ids": ids,
            "metadatas": metadatas or None
        }
        if self.embedding_function:
            embeddings = self.embedding_function.create_embeddings(documents)
            params["embeddings"] = embeddings
        self.collection.add(**params)
 
    def query_data(self, query_text=None, n_results=10):
        if not query_text:
            raise ValueError("Query text must be provided.")
        query_embeddings = self.embedding_function.create_embeddings(query_text)
        return self.collection.query(query_embeddings=query_embeddings, n_results=n_results)