#!/usr/bin/env python3
import asyncio
import json
import numpy as np
import time
from typing import List, Dict, Any

import sys
sys.path.append('.')

from scripts.async_processing import ParallelPromptGenerator, RateLimitTier
from scripts.inference import EmbeddingToPromptInference


def create_test_embeddings(count: int = 10) -> List[List[float]]:
    """Create random test embeddings for demonstration."""
    np.random.seed(42)  # For reproducible results
    embeddings = []
    for i in range(count):
        # Create slightly varied embeddings around different centers
        center = np.random.randn(1024) * 0.1
        noise = np.random.randn(1024) * 0.01
        embedding = center + noise
        embeddings.append(embedding.tolist())
    return embeddings


async def run_parallel_example():
    """Run parallel prompt generation example."""
    
    # Load configuration
    import yaml
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ config.yaml not found")
        return
    
    # Initialize inference engine
    model_path = "models/checkpoints/best_model.pt"
    try:
        inference_engine = EmbeddingToPromptInference(model_path)
        print("✅ Inference engine initialized")
    except Exception as e:
        print(f"❌ Failed to initialize inference engine: {e}")
        return
    
    # Create test embeddings
    test_embeddings = create_test_embeddings(10)
    print(f"✅ Created {len(test_embeddings)} test embeddings")
    
    # Initialize parallel generator
    try:
        generator = ParallelPromptGenerator(
            inference_engine=inference_engine,
            config=config,
            rate_limit_tier=RateLimitTier.TIER_2
        )
        print("✅ Parallel generator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize parallel generator: {e}")
        return
    
    try:
        # Run parallel processing
        print("🔄 Starting parallel processing...")
        start_time = time.time()
        
        results = await generator.generate_prompts_with_evaluation(
            test_embeddings,
            max_concurrent=5
        )
        
        processing_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if r.get('original_result', {}).get('success', False))
        
        print(f"✅ Processing completed in {processing_time:.2f}s")
        print(f"📊 Results: {successful}/{len(test_embeddings)} successful")
        
        # Show sample results
        if results:
            print("\n📝 Sample results:")
            for i, result in enumerate(results[:3]):
                if result.get('original_result', {}).get('success'):
                    prompt = result.get('original_prompt', '')
                    similarity = result.get('original_result', {}).get('similarity', 0)
                    print(f"  {i+1}. {prompt[:60]}... (similarity: {similarity:.3f})")
        
        # Performance metrics
        if successful > 0:
            avg_similarity = np.mean([
                r.get('original_result', {}).get('similarity', 0) 
                for r in results if r.get('original_result', {}).get('success', False)
            ])
            print(f"📈 Average similarity: {avg_similarity:.3f}")
            print(f"⚡ Processing rate: {len(test_embeddings)/processing_time:.1f} embeddings/sec")
        
    except Exception as e:
        print(f"❌ Parallel processing failed: {e}")
    
    finally:
        # Cleanup
        await generator.close()
        print("🧹 Cleanup completed")


def main():
    """Main function to run the parallel processing example."""
    print("🚀 Parallel Embedding-to-Prompt Generation Example")
    print("=" * 50)
    
    try:
        asyncio.run(run_parallel_example())
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✅ Example completed!")


if __name__ == "__main__":
    main() 