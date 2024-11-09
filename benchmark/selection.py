from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


class SelectionAlgorithm(ABC):
    """
    All solutions must implement this interface.
    """
    def __init__(self, agents: List[Dict[str, Any]], ids: List[str]) -> None:
        self.agents = agents
        self.ids = ids
        self.initialize(agents, ids)

    @abstractmethod
    def select(self, query: str, **kwargs) -> str:
        # TODO: Add agent selection logic here
        ...

    @abstractmethod
    def initialize(self, agents: List[Dict[str, Any]], ids: List[str]) -> str:
        # TODO: Initialize vectorstore
        ...


class ExampleAlgorithm(SelectionAlgorithm):
    def __init__(self, agents: List[Dict], agent_ids: List[str]):
        self.agents = agents
        self.agent_ids = agent_ids
    
    def select(self, query: str) -> str:
        # This is just a dummy implementation that always returns the first agent
        return self.agent_ids[0]


class FaissSelectionAlgorithm(SelectionAlgorithm):
    """
    FAISS-based selection algorithm with cross-encoder reranking.
    """
    def __init__(self, agents: List[Dict], agent_ids: List[str]):
        self.initialize(agents, agent_ids)

    def initialize(self, agents: List[Dict], agent_ids: List[str]) -> None:
        self.agents = agents
        self.ids = agent_ids
        
        # Changed to all-mpnet-base-v2 model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Create FAISS index
        agent_descriptions = [agent['description'] for agent in agents]
        embeddings = self.embedding_model.encode(agent_descriptions)
        
        dimension = embeddings.shape[1]  # Note: MPNet creates 768-dimensional embeddings
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))

    def select(self, query: str) -> Tuple[str, str]:
        # Get embeddings for the query
        query_embedding = self.embedding_model.encode([query])
        
        # Search FAISS index
        D, I = self.index.search(query_embedding, k=3)
        
        # Get candidate descriptions for reranking
        candidates = [self.agents[idx] for idx in I[0]]
        candidate_ids = [self.ids[idx] for idx in I[0]]
        
        # Debug logging
        print(f"\nQuery: {query}")
        print("Top candidates:")
        for i, (cand, score) in enumerate(zip(candidates, D[0])):
            print(f"{i+1}. {cand['name']} (distance: {score:.3f})")
        
        # Prepare pairs for cross-encoder
        pairs = [[query, cand['description']] for cand in candidates]
        
        # Get scores from cross-encoder
        scores = self.cross_encoder.predict(pairs)
        
        # Debug logging
        print("\nCross-encoder scores:")
        for i, (cand, score) in enumerate(zip(candidates, scores)):
            print(f"{cand['name']}: {score:.3f}")
        
        # Get best match
        best_idx = np.argmax(scores)
        selected_id = candidate_ids[best_idx]
        selected_name = candidates[best_idx]['name']
        
        return selected_id, selected_name
