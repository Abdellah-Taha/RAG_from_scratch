from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = PROJECT_ROOT / "cache" / "movie_embeddings.npy"
MOVIES_PATH = PROJECT_ROOT / "data" / "movies.json"

class SemanticSearch:
    def __init__(self):
        self.model =SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        movies_list = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movies_list.append(f"{doc['title']}: {doc['description']}")
        self.embedding = self.model.encode(movies_list, show_progress_bar=True)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE_PATH, self.embedding)
        return self.embedding
        
    def generate_embedding(self, text):
        if len("".join(text.split())) == 0:
            raise ValueError("empty text")
        embedding = self.model.encode([text])
        return embedding[0]
        

    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        if not CACHE_PATH.exists():
            return self.build_embeddings(documents)
        self.embedding = np.load(CACHE_PATH)
        if len(self.embedding) != len(documents):
            return self.build_embeddings(documents)
        return self.embedding
        
    def search(self, query, limit):
        if self.embedding is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        cosine_similarity_list = []

        for doc, doc_embedding in zip(self.documents, self.embedding):
            score = cosine_similarity(query_embedding, doc_embedding)
            cosine_similarity_list.append((score, doc))

        cosine_similarity_list.sort(key=lambda result: result[0], reverse=True)
        return [
            {
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            }
            for score, doc in cosine_similarity_list[:limit]
        ]

def verify_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")

def embed_text(text):
    semantic = SemanticSearch()
    embedding = semantic.generate_embedding(text)
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic = SemanticSearch()
    with MOVIES_PATH.open("rb") as f:
        documents = json.load(f)["movies"]
    embeddings = semantic.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semantic = SemanticSearch()
    embed_query = semantic.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embed_query[:3]}")
    print(f"Shape: {embed_query.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)