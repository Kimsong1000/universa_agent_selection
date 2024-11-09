from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BaseEmbeddingFunction:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return np.array(embeddings)

class CrossEncoderRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def rank(self, query: str, candidates: List[dict]) -> Tuple[Dict, float]:
        scores = []
        for agent in candidates:
            inputs = self.tokenizer(query, agent["description"], 
                                  return_tensors="pt", 
                                  truncation=True, 
                                  padding=True)
            outputs = self.model(**inputs)
            score = outputs.logits.item()
            scores.append((agent, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0]

class SelectionAlgorithm(ABC):
    @abstractmethod
    def select(self, query: str, **kwargs) -> Tuple[str, str]:
        pass

# Simple example algorithm that just returns the first agent
class ExampleAlgorithm(SelectionAlgorithm):
    def __init__(self, agents: List[Dict], agent_ids: List[str]):
        self.agents = agents
        self.ids = agent_ids
    
    def select(self, query: str, **kwargs) -> Tuple[str, str]:
        # Just return the first agent as an example
        return self.ids[0], self.agents[0]['name']

class FaissSelectionAlgorithm(SelectionAlgorithm):
    def __init__(self, agents: List[Dict], agent_ids: List[str]):
        self.agents = agents
        self.ids = agent_ids
        self.embedding_function = BaseEmbeddingFunction()
        self.cross_encoder = CrossEncoderRanker()
        
        descriptions = [agent["description"] for agent in agents]
        embeddings = self.embedding_function.create_embeddings(descriptions)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def select(self, query: str, **kwargs) -> Tuple[str, str]:
        query_embedding = self.embedding_function.create_embeddings([query])
        k = 3
        D, I = self.index.search(query_embedding, k)
        
        candidates = [self.agents[idx] for idx in I[0]]
        candidate_ids = [self.ids[idx] for idx in I[0]]
        
        if kwargs.get('verbose', False):
            print(f"\nQuery: {query}")
            print("FAISS Top candidates:")
            for i, (cand, score) in enumerate(zip(candidates, D[0])):
                print(f"{i+1}. {cand['name']} (distance: {score:.3f})")
        
        best_agent, best_score = self.cross_encoder.rank(query, candidates)
        best_idx = candidates.index(best_agent)
        selected_id = candidate_ids[best_idx]
        
        if kwargs.get('verbose', False):
            print("\nCross-encoder final selection:")
            print(f"Selected: {best_agent['name']} (score: {best_score:.3f})")
        
        return selected_id, best_agent['name']
