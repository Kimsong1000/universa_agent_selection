import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

# Define the embedding function using all-mpnet-base-v2
class BaseEmbeddingFunction:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        # Generate embeddings and convert to a numpy array
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
    # Generate embeddings for each agent's description
    descriptions = [agent["description"] for agent in agents]
    embeddings = embedding_function.create_embeddings(descriptions)
    
    # Initialize FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance
    faiss_index.add(embeddings)  # Add agent embeddings to the FAISS index
    
    return faiss_index, embeddings

# Run query and find best matching agent using FAISS
def run_queries(faiss_index, agents, embedding_function):
    from queries import QUERIES  # Import QUERIES from the queries.py file
    
    for user_prompt in QUERIES:
        print(f"Query: '{user_prompt}'")
        
        # Create embedding for the query
        query_embedding = embedding_function.create_embeddings([user_prompt])
        
        # Perform the search on the FAISS index
        distances, indices = faiss_index.search(query_embedding, k=3)  # Get top 3 matches
        
        # Print top 3 matches with just the name and distance
        for i, idx in enumerate(indices[0]):
            agent = agents[idx]
            print(f"Match {i + 1}: Name: {agent['name']}, Distance: {distances[0][i]}")
        
        # Print the best match (first in the list)
        best_match_index = indices[0][0]
        best_agent = agents[best_match_index]
        best_distance = distances[0][0]
        print(f"Best Match: Name: {best_agent['name']}, Distance: {best_distance}\n")

if __name__ == "__main__":
    # Load agents and initialize embedding function
    agents_folder = "C:\\Users\\kimso\\Desktop\\universa\\My_example\\agents"
    agents = load_agents_from_folder(agents_folder)
    embedding_function = BaseEmbeddingFunction("all-mpnet-base-v2")
    
    # Setup FAISS index with agent embeddings
    faiss_index, embeddings = setup_faiss_index(agents, embedding_function)
    
    # Run queries to find the best matching agents
    run_queries(faiss_index, agents, embedding_function)

    # Load agents and initialize embedding function
    agents_folder = "C:\\Users\\kimso\\Desktop\\universa\\My_example\\agents"
    agents = load_agents_from_folder(agents_folder)
    embedding_function = BaseEmbeddingFunction("all-mpnet-base-v2")
    
    # Setup FAISS index with agent embeddings
    faiss_index, embeddings = setup_faiss_index(agents, embedding_function)
    
    # Run queries to find the best matching agents
    run_queries(faiss_index, agents, embedding_function)
