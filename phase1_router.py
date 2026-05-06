
"""
Phase 1: Vector-Based Persona Matching using ChromaDB and sentence-transformers
"""

import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from typing import List, Dict, Tuple

class PersonaRouter:
    """Routes posts to relevant bot personas using cosine similarity"""
    
    def __init__(self):
        # Initialize ChromaDB with sentence-transformers embeddings
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Using all-MiniLM-L6-v2 for good performance (free, local)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="bot_personas",
            embedding_function=self.embedding_fn
        )
        
        # Define bot personas
        self.personas = {
            "Bot A": {
                "description": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
                "personality": "tech_maximalist"
            },
            "Bot B": {
                "description": "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
                "personality": "doomer_skeptic"
            },
            "Bot C": {
                "description": "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI.",
                "personality": "finance_bro"
            }
        }
        
        self._setup_personas()
    
    def _setup_personas(self):
        """Store bot personas in vector database"""
        # Clear existing data
        try:
            self.collection.delete(ids=list(self.personas.keys()))
        except:
            pass
        
        # Add personas to collection
        self.collection.add(
            ids=list(self.personas.keys()),
            documents=[p["description"] for p in self.personas.values()],
            metadatas=[
                {"personality": p["personality"], "bot_id": bot_id}
                for bot_id, p in self.personas.items()
            ]
        )
        print(f"✅ Loaded {len(self.personas)} bot personas into vector DB\n")
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def route_post_to_bots(self, post_content: str, threshold: float = 0.65) -> List[Dict]:
        """
        Route a post to relevant bots based on vector similarity
        
        Args:
            post_content: The post text to analyze
            threshold: Minimum cosine similarity (adjusted to 0.65 for realistic results)
        
        Returns:
            List of matched bots with similarity scores
        """
        print(f"\n📝 Analyzing Post: \"{post_content}\"")
        print(f"Threshold: {threshold}\n")
        
        # Generate embedding for the post
        post_embedding = self.embedding_fn([post_content])[0]
        
        # Get all personas and their embeddings
        results = self.collection.get(include=["documents", "embeddings", "metadatas"])
        
        matches = []
        for i, bot_id in enumerate(results["ids"]):
            persona_embedding = results["embeddings"][i]
            similarity = self.cosine_similarity(post_embedding, persona_embedding)
            
            if similarity > threshold:
                matches.append({
                    "bot_id": bot_id,
                    "personality": results["metadatas"][i]["personality"],
                    "similarity_score": round(similarity, 4),
                    "persona_description": results["documents"][i][:100] + "..."
                })
        
        # Sort by similarity score descending
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Print results
        print("🎯 Routing Results:")
        if matches:
            for match in matches:
                print(f"  ✓ {match['bot_id']} - {match['personality']} "
                      f"(similarity: {match['similarity_score']:.4f})")
        else:
            print("  ✗ No bots matched the post content")
        
        return matches


def run_phase1_demo():
    """Demo Phase 1 functionality"""
    print("=" * 60)
    print("PHASE 1: Vector-Based Persona Matching")
    print("=" * 60)
    
    router = PersonaRouter()
    
    # Test posts
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin reaches new all-time high as institutional investors pile in",
        "Climate change is accelerating due to unchecked industrial growth",
        "I love hiking and disconnecting from technology on weekends"
    ]
    
    results = {}
    for post in test_posts:
        matches = router.route_post_to_bots(post, threshold=0.65)
        results[post] = matches
        print()
    
    return results


if __name__ == "__main__":
    run_phase1_demo()
