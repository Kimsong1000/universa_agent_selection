import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List

# Define the embedding function using SentenceTransformer
class BaseEmbeddingFunction:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return np.array(embeddings)

# Load agents from JSON files
def load_agents_from_folder(folder_path):
    agents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r") as file:
                data = json.load(file)
                agents.append(data)
    return agents

# Initialize FAISS index and load agents
def setup_faiss_index(agents, embedding_function):
    descriptions = [agent["description"] for agent in agents]
    embeddings = embedding_function.create_embeddings(descriptions)
    
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    
    return faiss_index, embeddings

# Cross-encoder class for re-ranking
class CrossEncoderRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def rank(self, query: str, candidates: List[dict]):
        scores = []
        for agent in candidates:
            inputs = self.tokenizer(query, agent["description"], return_tensors="pt", truncation=True, padding=True)
            outputs = self.model(**inputs)
            score = outputs.logits.item()  # Get the similarity score from the model
            scores.append((agent, score))
        
        # Sort candidates by cross-encoder scores in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0]  # Return the highest-ranked agent

# Run queries with FAISS and re-rank with Cross-encoder
def run_queries(faiss_index, agents, embedding_function, cross_encoder_ranker):
    from queries import QUERIES  # Import QUERIES from the queries.py file
    
    for user_prompt in QUERIES:
        print(f"Query: '{user_prompt}'")
        
        # Create embedding for the query
        query_embedding = embedding_function.create_embeddings([user_prompt])
        
        # Retrieve top 3 candidates using FAISS
        distances, indices = faiss_index.search(query_embedding, k=3)  # Get top 3 matches
        
        # Gather the top candidates based on FAISS retrieval
        top_candidates = []
        for i, idx in enumerate(indices[0]):
            agent = agents[idx]
            faiss_distance = distances[0][i]
            top_candidates.append(agent)
            print(f"FAISS Top {i + 1}: Name: {agent['name']}, Distance: {faiss_distance}")
        
        # Re-rank the top 3 candidates using the cross-encoder
        best_agent, best_score = cross_encoder_ranker.rank(user_prompt, top_candidates)
        
        # Print the best match based on cross-encoder ranking
        print(f"Best Match: Name: {best_agent['name']}, Cross-encoder Score: {best_score}\n")
    from queries import QUERIES  # Import QUERIES from the queries.py file
    
if __name__ == "__main__":
    # Load agents and initialize embedding function
    agents_folder = "C:\\Users\\kimso\\Desktop\\universa\\My_example\\agents"
    agents = load_agents_from_folder(agents_folder)
    embedding_function = BaseEmbeddingFunction("all-mpnet-base-v2")
    
    # Setup FAISS index with agent embeddings
    faiss_index, embeddings = setup_faiss_index(agents, embedding_function)
    
    # Initialize cross-encoder ranker
    cross_encoder_ranker = CrossEncoderRanker()
    
    # Run queries to find the best matching agents
    run_queries(faiss_index, agents, embedding_function, cross_encoder_ranker)
