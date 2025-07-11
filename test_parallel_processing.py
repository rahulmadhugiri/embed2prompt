#!/usr/bin/env python3
"""
Test Script for Parallel Embedding-to-Prompt Processing
Demonstrates the new parallel processing capabilities with 10 examples,
including rate limiting, quality evaluation, and performance metrics.
"""

import asyncio
import json
import time
import requests
import numpy as np
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Test configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
API_KEY = os.getenv('API_KEY', None)  # Optional API key


class ParallelProcessingTester:
    """Test class for parallel processing capabilities."""
    
    def __init__(self, base_url: str = API_BASE_URL, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
        
    def generate_test_embeddings(self, num_samples: int = 10) -> List[List[float]]:
        """Generate test embeddings for the demo."""
        logger.info(f"Generating {num_samples} test embeddings...")
        
        # Create diverse embeddings with different patterns
        embeddings = []
        
        # Pattern 1: Clustered embeddings (similar to each other)
        base_embedding = np.random.random(1024)
        for i in range(3):
            variation = base_embedding + np.random.normal(0, 0.1, 1024)
            embeddings.append(variation.tolist())
        
        # Pattern 2: Diverse embeddings (different domains)
        for i in range(4):
            diverse_embedding = np.random.random(1024)
            # Add some structure to make them more realistic
            diverse_embedding[i*100:(i+1)*100] *= 2  # Emphasize different regions
            embeddings.append(diverse_embedding.tolist())
        
        # Pattern 3: Sparse embeddings (mostly zeros with some peaks)
        for i in range(3):
            sparse_embedding = np.zeros(1024)
            # Add some random peaks
            indices = np.random.choice(1024, size=50, replace=False)
            sparse_embedding[indices] = np.random.random(50)
            embeddings.append(sparse_embedding.tolist())
        
        logger.info(f"Generated {len(embeddings)} test embeddings")
        return embeddings
    
    def test_api_health(self) -> bool:
        """Test if the API is healthy and ready."""
        try:
            response = requests.get(f"{self.base_url}/health", headers=self.headers)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"API Health: {health_data['status']}")
                logger.info(f"Model loaded: {health_data['model_loaded']}")
                return health_data['model_loaded']
            else:
                logger.error(f"Health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to API: {e}")
            return False
    
    def test_parallel_processing(self, embeddings: List[List[float]], 
                                max_concurrent: int = 5,
                                rate_limit_tier: str = "tier_2",
                                evaluate_quality: bool = True) -> Dict[str, Any]:
        """Test the parallel processing endpoint."""
        
        logger.info(f"Testing parallel processing with {len(embeddings)} embeddings")
        logger.info(f"Max concurrent: {max_concurrent}")
        logger.info(f"Rate limit tier: {rate_limit_tier}")
        logger.info(f"Quality evaluation: {evaluate_quality}")
        
        payload = {
            "embeddings": embeddings,
            "max_concurrent": max_concurrent,
            "rate_limit_tier": rate_limit_tier,
            "evaluate_quality": evaluate_quality,
            "generate_variants": False,  # Keep it simple for the demo
            "return_metadata": True
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/generate_prompts_parallel",
                json=payload,
                headers=self.headers,
                timeout=300  # 5 minute timeout
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Add our timing to the result
                result['client_processing_time'] = processing_time
                
                logger.info(f"✅ Parallel processing completed in {processing_time:.2f}s")
                logger.info(f"Server processing time: {result['summary']['processing_time']:.2f}s")
                
                return result
            else:
                logger.error(f"❌ Parallel processing failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return {"error": f"HTTP {response.status_code}", "message": response.text}
                
        except Exception as e:
            logger.error(f"❌ Request failed: {e}")
            return {"error": str(e)}
    
    def test_sequential_processing(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """Test sequential processing for comparison."""
        
        logger.info(f"Testing sequential processing with {len(embeddings)} embeddings")
        
        start_time = time.time()
        results = []
        
        for i, embedding in enumerate(embeddings):
            try:
                payload = {
                    "target_embedding": embedding,
                    "return_metadata": True
                }
                
                response = requests.post(
                    f"{self.base_url}/generate_prompt",
                    json=payload,
                    headers=self.headers,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "index": i,
                        "success": True,
                        "prompt": result["prompt"],
                        "metadata": result.get("metadata", {})
                    })
                else:
                    results.append({
                        "index": i,
                        "success": False,
                        "error": f"HTTP {response.status_code}"
                    })
                    
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Sequential processing completed in {processing_time:.2f}s")
        
        return {
            "results": results,
            "processing_time": processing_time,
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"])
        }
    
    def compare_performance(self, parallel_result: Dict[str, Any], 
                           sequential_result: Dict[str, Any]) -> None:
        """Compare performance between parallel and sequential processing."""
        
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        
        if "error" in parallel_result:
            print(f"❌ Parallel processing failed: {parallel_result['error']}")
            return
        
        # Extract timing data
        parallel_time = parallel_result.get('client_processing_time', 0)
        sequential_time = sequential_result.get('processing_time', 0)
        
        # Extract success rates
        parallel_success = parallel_result['summary']['successful']
        parallel_total = parallel_result['summary']['total_processed']
        sequential_success = sequential_result['successful']
        sequential_total = len(sequential_result['results'])
        
        print(f"📊 Processing Time:")
        print(f"  Parallel:   {parallel_time:.2f}s")
        print(f"  Sequential: {sequential_time:.2f}s")
        print(f"  Speedup:    {sequential_time/parallel_time:.1f}x faster" if parallel_time > 0 else "  Speedup:    N/A")
        
        print(f"\n📈 Success Rate:")
        print(f"  Parallel:   {parallel_success}/{parallel_total} ({parallel_success/parallel_total*100:.1f}%)")
        print(f"  Sequential: {sequential_success}/{sequential_total} ({sequential_success/sequential_total*100:.1f}%)")
        
        # Show parallel processing details
        if 'metadata' in parallel_result:
            metadata = parallel_result['metadata']
            print(f"\n🔧 Parallel Processing Details:")
            print(f"  Rate Limiting: {metadata['processing_details']['rate_limiting']}")
            print(f"  Quality Eval:  {metadata['processing_details']['quality_evaluation']}")
            print(f"  Req/sec:       {metadata['performance_metrics']['requests_per_second']:.2f}")
            print(f"  Avg time:      {metadata['performance_metrics']['average_processing_time']:.2f}s")
        
        # Show quality metrics if available
        if parallel_result['summary'].get('average_similarity') is not None:
            print(f"\n🎯 Quality Metrics:")
            print(f"  Avg Similarity: {parallel_result['summary']['average_similarity']:.3f}")
            print(f"  Max Similarity: {parallel_result['summary']['max_similarity']:.3f}")
            print(f"  Min Similarity: {parallel_result['summary']['min_similarity']:.3f}")
    
    def display_sample_results(self, parallel_result: Dict[str, Any], num_samples: int = 3) -> None:
        """Display a few sample results."""
        
        if "error" in parallel_result:
            return
        
        print("\n" + "="*60)
        print("SAMPLE RESULTS")
        print("="*60)
        
        results = parallel_result['results'][:num_samples]
        
        for i, result in enumerate(results):
            print(f"\n📝 Sample {i+1}:")
            print(f"  Prompt: {result['original_prompt'][:100]}...")
            
            if result['original_result'].get('similarity') is not None:
                similarity = result['original_result']['similarity']
                print(f"  Quality Score: {similarity:.3f}")
                print(f"  Success: {'✅' if result['original_result']['success'] else '❌'}")
            
            if result['original_result'].get('generated_text'):
                generated_text = result['original_result']['generated_text']
                print(f"  Generated: {generated_text[:100]}...")


def main():
    """Main function to run the parallel processing test."""
    
    print("🚀 Parallel Embedding-to-Prompt Processing Test")
    print("="*60)
    
    # Initialize tester
    tester = ParallelProcessingTester()
    
    # Check API health
    if not tester.test_api_health():
        print("❌ API is not healthy. Please start the server first.")
        return
    
    # Generate test embeddings
    embeddings = tester.generate_test_embeddings(10)
    
    # Test parallel processing
    print("\n🔄 Testing Parallel Processing...")
    parallel_result = tester.test_parallel_processing(
        embeddings,
        max_concurrent=5,
        rate_limit_tier="tier_2",
        evaluate_quality=True
    )
    
    # Test sequential processing for comparison
    print("\n🔄 Testing Sequential Processing...")
    sequential_result = tester.test_sequential_processing(embeddings)
    
    # Compare performance
    tester.compare_performance(parallel_result, sequential_result)
    
    # Display sample results
    tester.display_sample_results(parallel_result)
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    main() 