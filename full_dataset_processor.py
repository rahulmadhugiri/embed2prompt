#!/usr/bin/env python3
"""
Full Dataset Processor for Alpaca Dataset
Processes all 52,002 datapoints with parallel embedding generation,
local caching, and Pinecone upload with proper metadata format.
"""

import csv
import json
import time
import os
import threading
import requests
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pinecone imports
try:
    from pinecone import Pinecone
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False
    print("⚠️  Pinecone not available. Please install with: pip install pinecone-client")


class FullDatasetProcessor:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the full dataset processor."""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # OpenAI API configuration
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = self.config['openai']['embedding_model']
        self.dimensions = self.config['openai']['embedding_dimensions']
        self.url = "https://api.openai.com/v1/embeddings"
        
        # Pinecone configuration
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_index_name = self.config['pinecone']['index_name']
        
        # Processing configuration
        self.batch_size = 200  # Process in batches of 200
        self.max_workers = 10  # More workers for full dataset
        self.lock = threading.Lock()
        
        # Cache configuration
        self.cache_dir = Path("data/embeddings_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "full_dataset_embeddings.json"
        self.progress_file = self.cache_dir / "processing_progress.json"
        
        # Validation
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if HAS_PINECONE and not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        print("✅ Full dataset processor initialized")
        print(f"   Model: {self.model}")
        print(f"   Dimensions: {self.dimensions}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Max workers: {self.max_workers}")
        print(f"   Cache directory: {self.cache_dir}")
    
    def clear_pinecone_database(self) -> bool:
        """Clear all vectors from the Pinecone database."""
        if not HAS_PINECONE:
            print("❌ Pinecone not available, cannot clear database")
            return False
        
        try:
            print("🗑️  Clearing Pinecone database...")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=self.pinecone_api_key)
            index = pc.Index(self.pinecone_index_name)
            
            # Delete all vectors
            index.delete(delete_all=True)
            
            print("✅ Pinecone database cleared successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing Pinecone database: {e}")
            return False
    
    def load_progress(self) -> Dict[str, Any]:
        """Load processing progress from cache."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"last_processed_index": -1, "total_processed": 0, "failed_indices": []}
    
    def save_progress(self, progress: Dict[str, Any]):
        """Save processing progress to cache."""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def load_cached_embeddings(self) -> Dict[str, Any]:
        """Load cached embeddings from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def save_cached_embeddings(self, embeddings: Dict[str, Any]):
        """Save embeddings to cache."""
        with open(self.cache_file, 'w') as f:
            json.dump(embeddings, f, indent=2)
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count for a text (rough approximation)."""
        # Simple approximation: ~0.75 tokens per word
        return max(1, int(len(text.split()) * 0.75))
    
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
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        "index": index,
                        "text": text,
                        "embedding": None,
                        "tokens_used": 0,
                        "success": False,
                        "error": str(e)
                    }
    
    def generate_embeddings_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for a batch of texts in parallel."""
        results = [None] * len(batch_data)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.generate_single_embedding, item["output"], item["original_index"]): i
                for i, item in enumerate(batch_data)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                batch_index = future_to_index[future]
                try:
                    result = future.result()
                    results[batch_index] = result
                except Exception as e:
                    results[batch_index] = {
                        "index": batch_data[batch_index]["original_index"],
                        "text": batch_data[batch_index]["output"],
                        "embedding": None,
                        "tokens_used": 0,
                        "success": False,
                        "error": str(e)
                    }
        
        return results
    
    def load_full_dataset(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load the full alpaca dataset."""
        print(f"📖 Loading full dataset from {csv_path}...")
        
        data_points = []
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for i, row in enumerate(reader):
                data_points.append({
                    "original_index": i,
                    "prompt": str(row.get('prompt', '')),
                    "output": str(row.get('output', ''))
                })
        
        print(f"✅ Dataset loaded with {len(data_points):,} rows")
        return data_points
    
    def upload_batch_to_pinecone(self, batch_embeddings: List[Dict[str, Any]], 
                                batch_data: List[Dict[str, Any]]) -> bool:
        """Upload a batch of embeddings to Pinecone."""
        if not HAS_PINECONE:
            return False
        
        try:
            # Initialize Pinecone
            pc = Pinecone(api_key=self.pinecone_api_key)
            index = pc.Index(self.pinecone_index_name)
            
            # Prepare vectors for upload
            vectors = []
            for embedding_result, original_data in zip(batch_embeddings, batch_data):
                if embedding_result["success"]:
                    vector = {
                        'id': f"alpaca-{embedding_result['index']}",
                        'values': embedding_result['embedding'],
                        'metadata': {
                            'output': original_data['output'],
                            'prompt': original_data['prompt'],
                            'tokens': self.count_tokens(original_data['output']),
                            'type': 'alpaca'
                        }
                    }
                    vectors.append(vector)
            
            # Upload batch
            if vectors:
                index.upsert(vectors)
                return True
            
        except Exception as e:
            print(f"❌ Error uploading batch to Pinecone: {e}")
            return False
        
        return False
    
    def process_full_dataset(self, csv_path: str, resume: bool = True) -> Dict[str, Any]:
        """Process the full dataset with parallel embedding generation."""
        print("🚀 Starting full dataset processing...")
        print("=" * 80)
        
        # Load dataset
        data_points = self.load_full_dataset(csv_path)
        total_rows = len(data_points)
        
        # Load progress and cached embeddings
        progress = self.load_progress() if resume else {"last_processed_index": -1, "total_processed": 0, "failed_indices": []}
        cached_embeddings = self.load_cached_embeddings() if resume else {}
        
        # Calculate batches
        total_batches = math.ceil(total_rows / self.batch_size)
        start_batch = (progress["last_processed_index"] + 1) // self.batch_size
        
        # Calculate remaining work
        remaining_rows = total_rows - (progress["last_processed_index"] + 1)
        remaining_batches = total_batches - start_batch
        
        print(f"📊 Processing Configuration:")
        print(f"   Total rows: {total_rows:,}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Total batches: {total_batches}")
        print(f"   Starting from batch: {start_batch + 1}")
        print(f"   Remaining batches: {remaining_batches}")
        print(f"   Previously processed: {progress['total_processed']:,}")
        print(f"   Remaining to process: {remaining_rows:,}")
        
        if resume and start_batch > 0:
            print(f"   📂 Resuming from batch {start_batch + 1}/{total_batches}")
            print(f"   ⏩ Skipping {start_batch} completed batches")
        
        # Processing statistics
        start_time = time.time()
        total_tokens = 0
        successful_embeddings = progress['total_processed']  # Start with existing count
        failed_embeddings = len(progress['failed_indices'])
        
        # Process in batches
        for batch_num in range(start_batch, total_batches):
            batch_start = batch_num * self.batch_size
            batch_end = min(batch_start + self.batch_size, total_rows)
            batch_data = data_points[batch_start:batch_end]
            
            print(f"\n🔄 Processing batch {batch_num + 1}/{total_batches} (rows {batch_start:,}-{batch_end:,})")
            
            # Generate embeddings for this batch
            batch_start_time = time.time()
            batch_results = self.generate_embeddings_batch(batch_data)
            batch_time = time.time() - batch_start_time
            
            # Process results
            batch_successful = 0
            batch_failed = 0
            batch_tokens = 0
            
            for i, result in enumerate(batch_results):
                original_index = result["index"]
                
                if result["success"]:
                    batch_successful += 1
                    batch_tokens += result["tokens_used"]
                    
                    # Cache the embedding
                    cached_embeddings[str(original_index)] = {
                        "embedding": result["embedding"],
                        "tokens_used": result["tokens_used"],
                        "prompt": batch_data[i]["prompt"],
                        "output": batch_data[i]["output"]
                    }
                else:
                    batch_failed += 1
                    progress["failed_indices"].append(original_index)
            
            # Upload to Pinecone
            pinecone_success = self.upload_batch_to_pinecone(batch_results, batch_data)
            
            # Update statistics
            successful_embeddings += batch_successful
            failed_embeddings += batch_failed
            total_tokens += batch_tokens
            
            # Update progress
            progress["last_processed_index"] = batch_end - 1
            progress["total_processed"] = successful_embeddings
            
            # Save progress and cache
            self.save_progress(progress)
            self.save_cached_embeddings(cached_embeddings)
            
            # Print batch summary
            print(f"   ✅ Batch completed in {batch_time:.2f}s")
            print(f"   📊 Successful: {batch_successful}/{len(batch_data)}")
            print(f"   🔢 Tokens used: {batch_tokens:,}")
            print(f"   📤 Pinecone upload: {'✅' if pinecone_success else '❌'}")
            
            # Rate limiting delay
            time.sleep(1)
        
        # Final summary
        total_time = time.time() - start_time
        estimated_cost = (total_tokens / 1_000_000) * 0.02
        
        summary = {
            "total_processed": len(data_points),
            "successful_embeddings": successful_embeddings,
            "failed_embeddings": failed_embeddings,
            "total_tokens": total_tokens,
            "estimated_cost": estimated_cost,
            "processing_time": total_time,
            "average_time_per_embedding": total_time / successful_embeddings if successful_embeddings > 0 else 0
        }
        
        print("\n" + "=" * 80)
        print("🎉 FULL DATASET PROCESSING COMPLETED!")
        print("=" * 80)
        print(f"📊 Final Statistics:")
        print(f"   Total processed: {summary['total_processed']:,}")
        print(f"   Successful embeddings: {summary['successful_embeddings']:,}")
        print(f"   Failed embeddings: {summary['failed_embeddings']:,}")
        print(f"   Success rate: {(summary['successful_embeddings']/summary['total_processed']*100):.1f}%")
        print(f"   Total tokens used: {summary['total_tokens']:,}")
        print(f"   Estimated cost: ${summary['estimated_cost']:.4f}")
        print(f"   Total processing time: {summary['processing_time']/60:.1f} minutes")
        print(f"   Average time per embedding: {summary['average_time_per_embedding']:.3f}s")
        
        return summary


def main():
    """Main function to run the full dataset processing."""
    print("🚀 Full Alpaca Dataset Processing")
    print("=" * 80)
    
    # Check API keys
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment variables")
        return
    
    if not os.getenv('PINECONE_API_KEY'):
        print("❌ PINECONE_API_KEY not found in environment variables")
        return
    
    try:
        # Initialize processor
        processor = FullDatasetProcessor()
        
        # Check for existing progress
        progress = processor.load_progress()
        has_existing_progress = progress["total_processed"] > 0
        
        if has_existing_progress:
            print(f"\n📋 Existing progress detected!")
            print(f"   Previously processed: {progress['total_processed']:,} embeddings")
            print(f"   Last processed index: {progress['last_processed_index']:,}")
            print(f"   Failed indices: {len(progress['failed_indices'])}")
            
            # Calculate resume batch
            total_rows = 52002  # Known dataset size
            batch_size = 200
            start_batch = (progress["last_processed_index"] + 1) // batch_size
            total_batches = math.ceil(total_rows / batch_size)
            
            print(f"   Will resume from batch: {start_batch + 1}/{total_batches}")
            
            print("\n🔄 Choose an option:")
            print("1. Resume from where you left off (recommended)")
            print("2. Start fresh (clears all progress and Pinecone data)")
            
            choice = input("Enter your choice (1 or 2): ").strip()
            
            if choice == "1":
                print("✅ Resuming from existing progress...")
                resume_mode = True
                clear_database = False
            elif choice == "2":
                print("⚠️  Starting fresh - all progress will be lost!")
                confirm = input("Type 'YES' to confirm: ")
                if confirm == 'YES':
                    resume_mode = False
                    clear_database = True
                else:
                    print("❌ Operation cancelled.")
                    return
            else:
                print("❌ Invalid choice. Exiting.")
                return
        else:
            print("\n🆕 No existing progress found. Starting fresh.")
            print("\n⚠️  WARNING: This will clear your entire Pinecone database!")
            confirm = input("Type 'YES' to confirm clearing Pinecone database: ")
            
            if confirm == 'YES':
                resume_mode = False
                clear_database = True
            else:
                print("❌ Database clear not confirmed. Exiting.")
                return
        
        # Clear database if needed
        if clear_database:
            if not processor.clear_pinecone_database():
                print("❌ Failed to clear Pinecone database. Exiting.")
                return
        
        # Process the full dataset
        dataset_path = "data/alpaca_data.csv"
        summary = processor.process_full_dataset(dataset_path, resume=resume_mode)
        
        print(f"\n💾 Results cached in: {processor.cache_file}")
        print(f"📈 Progress saved in: {processor.progress_file}")
        
    except FileNotFoundError:
        print(f"❌ Dataset file not found: data/alpaca_data.csv")
        print("Please ensure the alpaca dataset is available")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 