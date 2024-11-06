from typing import List
from sentence_transformers import SentenceTransformer

class BaseEmbeddingFunction:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        # Load the SentenceTransformer model
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Use the model to create embeddings
        embeddings = self.model.encode(texts, convert_to_tensor=False)  # Convert to list of lists
        return embeddings

    def calculate_similarities(self, texts: List[str]) -> List[List[float]]:
        # Get embeddings
        embeddings = self.create_embeddings(texts)
        # Calculate similarity matrix
        similarities = [[self.model.similarity(emb1, emb2) for emb2 in embeddings] for emb1 in embeddings]
        return similarities
