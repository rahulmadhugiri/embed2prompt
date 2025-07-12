#!/usr/bin/env python3
import csv
import json
import time
import os
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ParallelEmbeddingGenerator:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = "text-embedding-3-small"
        self.dimensions = 1024
        self.url = "https://api.openai.com/v1/embeddings"
        self.lock = threading.Lock()
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
    def generate_single_embedding(self, text: str, index: int) -> Dict[str, Any]:
        """Generate embedding for a single text with retry logic."""
        max_retries = 3
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": text,
            "model": self.model,
            "dimensions": self.dimensions
        }
        
        for attempt in range(max_retries):
            try:
                with self.lock:
                    print(f"  Processing item {index + 1}/10...")
                
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    embedding = data["data"][0]["embedding"]
                    tokens_used = data["usage"]["total_tokens"]
                    
                    return {
                        "index": index,
                        "text": text,
                        "embedding": embedding,
                        "tokens_used": tokens_used,
                        "success": True
                    }
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    else:
                        with self.lock:
                            print(f"  ❌ Error processing item {index + 1}: {error_msg}")
                        return {
                            "index": index,
                            "text": text,
                            "embedding": None,
                            "tokens_used": 0,
                            "success": False,
                            "error": error_msg
                        }
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    with self.lock:
                        print(f"  ❌ Error processing item {index + 1}: {e}")
                    return {
                        "index": index,
                        "text": text,
                        "embedding": None,
                        "tokens_used": 0,
                        "success": False,
                        "error": str(e)
                    }
    
    def generate_embeddings_parallel(self, texts: List[str], max_workers: int = 5) -> List[Dict[str, Any]]:
        """Generate embeddings for multiple texts in parallel using ThreadPoolExecutor."""
        print(f"🔄 Generating embeddings for {len(texts)} texts (max {max_workers} workers)...")
        
        results = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.generate_single_embedding, text, i): i 
                for i, text in enumerate(texts)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    results[index] = {
                        "index": index,
                        "text": texts[index],
                        "embedding": None,
                        "tokens_used": 0,
                        "success": False,
                        "error": str(e)
                    }
        
        return results


def load_middle_data_points(csv_path: str, num_points: int = 10) -> List[Dict[str, str]]:
    """Load data points from the middle of the dataset using csv module."""
    print(f"📖 Loading dataset from {csv_path}...")
    
    # First pass: count total rows
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        total_rows = sum(1 for row in reader) - 1  # Subtract 1 for header
    
    print(f"✅ Dataset loaded with {total_rows:,} rows")
    
    # Calculate middle range
    middle_start = (total_rows - num_points) // 2
    middle_end = middle_start + num_points
    
    print(f"📍 Extracting {num_points} data points from rows {middle_start:,} to {middle_end:,}")
    
    # Second pass: extract middle data points
    data_points = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for i, row in enumerate(reader):
            if middle_start <= i < middle_end:
                data_points.append({
                    "original_index": i,
                    "prompt": str(row.get('prompt', '')),
                    "output": str(row.get('output', ''))
                })
            elif i >= middle_end:
                break
    
    return data_points


def main():
    print("🚀 Parallel Embedding Generation for Alpaca Dataset")
    print("=" * 60)
    
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    try:
        # Load 10 data points from the middle of the dataset
        dataset_path = "data/alpaca_data.csv"
        data_points = load_middle_data_points(dataset_path, 10)
        
        # Extract outputs for embedding generation
        outputs = [point["output"] for point in data_points]
        
        print(f"\n📝 Sample outputs to be processed:")
        for i, output in enumerate(outputs[:3]):
            preview = output[:80] + "..." if len(output) > 80 else output
            print(f"  {i+1}. {preview}")
        if len(outputs) > 3:
            print(f"  ... and {len(outputs) - 3} more")
        
        # Initialize parallel embedding generator
        generator = ParallelEmbeddingGenerator()
        
        # Generate embeddings in parallel
        print(f"\n🔄 Starting parallel embedding generation...")
        start_time = time.time()
        
        results = generator.generate_embeddings_parallel(outputs, max_workers=5)
        
        processing_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        total_tokens = sum(r["tokens_used"] for r in successful)
        estimated_cost = (total_tokens / 1_000_000) * 0.02  # $0.02 per 1M tokens for text-embedding-3-small
        
        print(f"\n✅ Parallel processing completed!")
        print(f"📊 Results Summary:")
        print(f"   Total processed: {len(results)}")
        print(f"   Successful: {len(successful)}")
        print(f"   Failed: {len(failed)}")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Average time per embedding: {processing_time/len(results):.2f}s")
        print(f"   Total tokens used: {total_tokens:,}")
        print(f"   Estimated cost: ${estimated_cost:.6f}")
        
        # Show sample embedding info
        if successful:
            sample_embedding = successful[0]["embedding"]
            print(f"   Embedding dimensions: {len(sample_embedding)}")
            print(f"   Sample embedding range: [{min(sample_embedding):.4f}, {max(sample_embedding):.4f}]")
        
        # Save results
        output_file = "parallel_embeddings_results.json"
        
        # Prepare data for saving
        save_data = []
        for i, (data_point, result) in enumerate(zip(data_points, results)):
            save_data.append({
                "original_dataset_index": data_point["original_index"],
                "prompt": data_point["prompt"],
                "output": data_point["output"],
                "embedding": result["embedding"],
                "tokens_used": result["tokens_used"],
                "success": result["success"],
                "error": result.get("error")
            })
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"💾 Results saved to {output_file}")
        
        # Show any errors
        if failed:
            print(f"\n⚠️ Failed embeddings:")
            for fail in failed:
                print(f"   Index {fail['index']}: {fail['error']}")
    
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_path}")
        print("Please ensure the alpaca dataset is available at data/alpaca_data.csv")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 