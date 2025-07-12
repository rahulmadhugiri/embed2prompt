#!/usr/bin/env python3
import json
import numpy as np
from typing import List, Dict, Any


def load_test_embeddings(file_path: str = "test_embeddings.json") -> List[np.ndarray]:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        embeddings = []
        for item in data:
            if 'embedding' in item:
                embedding = np.array(item['embedding'])
                embeddings.append(embedding)
        
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return []


def validate_embedding_dimensions(embeddings: List[np.ndarray], expected_dim: int = 1024) -> Dict[str, Any]:
    if not embeddings:
        return {"valid": False, "error": "No embeddings loaded"}
    
    invalid_embeddings = []
    for i, emb in enumerate(embeddings):
        if emb.shape[0] != expected_dim:
            invalid_embeddings.append(i)
    
    return {
        "valid": len(invalid_embeddings) == 0,
        "total": len(embeddings),
        "invalid_count": len(invalid_embeddings),
        "invalid_indices": invalid_embeddings
    }


def get_embedding_stats(embeddings: List[np.ndarray]) -> Dict[str, float]:
    if not embeddings:
        return {}
    
    all_values = np.concatenate(embeddings)
    
    return {
        "mean": float(np.mean(all_values)),
        "std": float(np.std(all_values)),
        "min": float(np.min(all_values)),
        "max": float(np.max(all_values))
    }


def test_embedding_similarity(embeddings: List[np.ndarray], num_samples: int = 5) -> List[float]:
    if len(embeddings) < 2:
        return []
    
    similarities = []
    sample_indices = np.random.choice(len(embeddings), min(num_samples, len(embeddings)), replace=False)
    
    for i in range(len(sample_indices) - 1):
        emb1 = embeddings[sample_indices[i]]
        emb2 = embeddings[sample_indices[i + 1]]
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarities.append(float(similarity))
    
    return similarities


def main():
    print("Testing embeddings data...")
    
    # Load test embeddings
    embeddings = load_test_embeddings()
    if not embeddings:
        print("❌ Failed to load embeddings")
        return
    
    print(f"✅ Loaded {len(embeddings)} embeddings")
    
    # Validate dimensions
    validation = validate_embedding_dimensions(embeddings)
    if validation["valid"]:
        print("✅ All embeddings have correct dimensions")
    else:
        print(f"❌ {validation['invalid_count']} embeddings have incorrect dimensions")
    
    # Get statistics
    stats = get_embedding_stats(embeddings)
    print(f"📊 Stats - Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    
    # Test similarities
    similarities = test_embedding_similarity(embeddings)
    if similarities:
        avg_sim = np.mean(similarities)
        print(f"🔗 Average similarity: {avg_sim:.3f}")
    
    print("Testing completed!")


if __name__ == "__main__":
    main() 