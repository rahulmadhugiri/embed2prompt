#!/usr/bin/env python3
"""
Quick Example: Parallel Embedding-to-Prompt Processing
This script demonstrates the parallel processing capabilities with 10 examples.
"""

import requests
import json
import time
import numpy as np
from typing import List

# Configuration
API_URL = "http://localhost:8000"
NUM_EMBEDDINGS = 10


def generate_sample_embeddings(num_samples: int = 10) -> List[List[float]]:
    """Generate sample embeddings for demonstration."""
    print(f"🎲 Generating {num_samples} sample embeddings...")
    
    embeddings = []
    
    # Generate diverse embeddings
    for i in range(num_samples):
        # Create embeddings with different patterns
        if i < 3:
            # High values in first quarter
            embedding = np.random.random(1024) * 0.5
            embedding[:256] += 0.5
        elif i < 6:
            # High values in middle
            embedding = np.random.random(1024) * 0.5
            embedding[256:768] += 0.5
        else:
            # High values in last quarter
            embedding = np.random.random(1024) * 0.5
            embedding[768:] += 0.5
        
        embeddings.append(embedding.tolist())
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    return embeddings


def test_parallel_processing():
    """Test the parallel processing endpoint."""
    
    print("🚀 Parallel Embedding-to-Prompt Processing Example")
    print("=" * 60)
    
    # Check API health
    print("🔍 Checking API health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API Status: {health['status']}")
            print(f"✅ Model Loaded: {health['model_loaded']}")
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Please make sure the API server is running on localhost:8000")
        return
    
    # Generate sample embeddings
    embeddings = generate_sample_embeddings(NUM_EMBEDDINGS)
    
    # Test parallel processing
    print(f"\n🔄 Processing {NUM_EMBEDDINGS} embeddings in parallel...")
    
    payload = {
        "embeddings": embeddings,
        "max_concurrent": 5,
        "rate_limit_tier": "tier_2",
        "evaluate_quality": True,
        "generate_variants": False,
        "return_metadata": True
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_URL}/generate_prompts_parallel",
            json=payload,
            timeout=300
        )
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            print(f"✅ Processing completed in {processing_time:.2f}s")
            print(f"📊 Summary:")
            print(f"   Total processed: {result['summary']['total_processed']}")
            print(f"   Successful: {result['summary']['successful']}")
            print(f"   Failed: {result['summary']['failed']}")
            print(f"   Server time: {result['summary']['processing_time']:.2f}s")
            
            # Show quality metrics if available
            if result['summary'].get('average_similarity') is not None:
                print(f"   Avg similarity: {result['summary']['average_similarity']:.3f}")
                print(f"   Max similarity: {result['summary']['max_similarity']:.3f}")
                print(f"   Min similarity: {result['summary']['min_similarity']:.3f}")
            
            # Show sample results
            print(f"\n📝 Sample Results:")
            for i, result_item in enumerate(result['results'][:3]):
                print(f"\n   Sample {i+1}:")
                prompt = result_item['original_prompt'][:80] + "..." if len(result_item['original_prompt']) > 80 else result_item['original_prompt']
                print(f"   Prompt: {prompt}")
                
                if result_item['original_result'].get('similarity') is not None:
                    similarity = result_item['original_result']['similarity']
                    print(f"   Quality: {similarity:.3f}")
            
            # Show performance metrics
            if 'metadata' in result:
                perf = result['metadata']['performance_metrics']
                print(f"\n⚡ Performance Metrics:")
                print(f"   Requests/sec: {perf['requests_per_second']:.2f}")
                print(f"   Avg time per request: {perf['average_processing_time']:.2f}s")
                print(f"   Speedup from parallelization: ~{NUM_EMBEDDINGS/perf['average_processing_time']:.1f}x")
            
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")


def main():
    """Main function."""
    test_parallel_processing()


if __name__ == "__main__":
    main() 