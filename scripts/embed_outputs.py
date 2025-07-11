#!/usr/bin/env python3
"""
Embedding Generation Script
Generates embeddings for all outputs in the dataset using OpenAI's text-embedding-3-small model.
Caches embeddings locally and optionally uploads to Pinecone.
"""

import pandas as pd
import numpy as np
import json
import jsonlines
import os
from pathlib import Path
import yaml
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import hashlib
from dotenv import load_dotenv

# Optional imports
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("OpenAI library not available. Please install with: pip install openai")

try:
    from pinecone import Pinecone
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False
    print("Pinecone library not available. Please install with: pip install pinecone-client")

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("PyArrow not available. Parquet format will not be supported.")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_env_variables():
    """Load environment variables from .env file."""
    load_dotenv()


def create_text_hash(text: str) -> str:
    """Create a hash of the text for caching purposes."""
    return hashlib.md5(text.encode()).hexdigest()


def load_existing_embeddings(cache_file: str) -> Dict[str, np.ndarray]:
    """Load existing embeddings from cache file."""
    embeddings = {}
    
    if not os.path.exists(cache_file):
        return embeddings
    
    try:
        if cache_file.endswith('.jsonl'):
            with jsonlines.open(cache_file, 'r') as reader:
                for item in reader:
                    text_hash = item['text_hash']
                    embedding = np.array(item['embedding'])
                    embeddings[text_hash] = embedding
        elif cache_file.endswith('.parquet') and HAS_PARQUET:
            df = pd.read_parquet(cache_file)
            for _, row in df.iterrows():
                text_hash = row['text_hash']
                embedding = np.array(row['embedding'])
                embeddings[text_hash] = embedding
        else:
            print(f"Unsupported cache file format: {cache_file}")
            
    except Exception as e:
        print(f"Error loading cache file {cache_file}: {e}")
        return {}
    
    print(f"Loaded {len(embeddings)} cached embeddings from {cache_file}")
    return embeddings


def save_embeddings_cache(embeddings: Dict[str, np.ndarray], cache_file: str, 
                         texts: Dict[str, str], prompts: Dict[str, str]):
    """Save embeddings to cache file."""
    cache_dir = Path(cache_file).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if cache_file.endswith('.jsonl'):
        with jsonlines.open(cache_file, 'w') as writer:
            for text_hash, embedding in embeddings.items():
                item = {
                    'text_hash': text_hash,
                    'text': texts.get(text_hash, ''),
                    'prompt': prompts.get(text_hash, ''),
                    'embedding': embedding.tolist(),
                    'embedding_dim': len(embedding)
                }
                writer.write(item)
                
    elif cache_file.endswith('.parquet') and HAS_PARQUET:
        data = []
        for text_hash, embedding in embeddings.items():
            data.append({
                'text_hash': text_hash,
                'text': texts.get(text_hash, ''),
                'prompt': prompts.get(text_hash, ''),
                'embedding': embedding.tolist(),
                'embedding_dim': len(embedding)
            })
        
        df = pd.DataFrame(data)
        df.to_parquet(cache_file, index=False)
    
    else:
        print(f"Unsupported cache file format: {cache_file}")
        return
    
    print(f"Saved {len(embeddings)} embeddings to {cache_file}")


def generate_embedding_batch(client: OpenAI, texts: List[str], model: str, 
                           max_retries: int = 3, delay: float = 1.0) -> List[np.ndarray]:
    """Generate embeddings for a batch of texts with retry logic."""
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=texts,
                model=model
            )
            
            embeddings = []
            for embedding_data in response.data:
                embeddings.append(np.array(embedding_data.embedding))
            
            return embeddings
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"Failed to generate embeddings after {max_retries} attempts: {e}")
                raise


def generate_embeddings(df: pd.DataFrame, config: Dict[str, Any], 
                       cache_file: str, batch_size: int = 50) -> Dict[str, np.ndarray]:
    """Generate embeddings for all outputs in the dataset."""
    
    if not HAS_OPENAI:
        raise ImportError("OpenAI library is required for embedding generation")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Load existing embeddings
    existing_embeddings = load_existing_embeddings(cache_file)
    
    # Prepare texts and hashes
    texts_to_embed = []
    text_hashes = []
    text_hash_to_text = {}
    text_hash_to_prompt = {}
    
    for idx, row in df.iterrows():
        text = str(row['output'])
        prompt = str(row['prompt'])
        text_hash = create_text_hash(text)
        
        text_hash_to_text[text_hash] = text
        text_hash_to_prompt[text_hash] = prompt
        
        if text_hash not in existing_embeddings:
            texts_to_embed.append(text)
            text_hashes.append(text_hash)
    
    print(f"Need to generate embeddings for {len(texts_to_embed)} texts")
    print(f"Using existing embeddings for {len(existing_embeddings)} texts")
    
    # Generate embeddings in batches
    model = config['openai']['embedding_model']
    new_embeddings = {}
    
    if texts_to_embed:
        with tqdm(total=len(texts_to_embed), desc="Generating embeddings") as pbar:
            for i in range(0, len(texts_to_embed), batch_size):
                batch_texts = texts_to_embed[i:i + batch_size]
                batch_hashes = text_hashes[i:i + batch_size]
                
                try:
                    batch_embeddings = generate_embedding_batch(
                        client, batch_texts, model
                    )
                    
                    for text_hash, embedding in zip(batch_hashes, batch_embeddings):
                        new_embeddings[text_hash] = embedding
                        existing_embeddings[text_hash] = embedding
                    
                    pbar.update(len(batch_texts))
                    
                    # Add small delay to avoid rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Continue with next batch
                    pbar.update(len(batch_texts))
    
    # Save updated cache
    if new_embeddings:
        save_embeddings_cache(existing_embeddings, cache_file, 
                             text_hash_to_text, text_hash_to_prompt)
        print(f"Generated {len(new_embeddings)} new embeddings")
    
    return existing_embeddings


def create_training_data(df: pd.DataFrame, embeddings: Dict[str, np.ndarray], 
                        output_file: str) -> None:
    """Create training data pairs (prompt, output_embedding)."""
    
    training_data = []
    
    for idx, row in df.iterrows():
        text = str(row['output'])
        prompt = str(row['prompt'])
        text_hash = create_text_hash(text)
        
        if text_hash in embeddings:
            training_data.append({
                'index': idx,
                'prompt': prompt,
                'output': text,
                'text_hash': text_hash,
                'embedding': embeddings[text_hash].tolist(),
                'embedding_dim': len(embeddings[text_hash])
            })
    
    # Save training data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_file.endswith('.jsonl'):
        with jsonlines.open(output_file, 'w') as writer:
            for item in training_data:
                writer.write(item)
    elif output_file.endswith('.parquet') and HAS_PARQUET:
        df_train = pd.DataFrame(training_data)
        df_train.to_parquet(output_file, index=False)
    else:
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
    
    print(f"Saved {len(training_data)} training samples to {output_file}")


def upload_to_pinecone(embeddings: Dict[str, np.ndarray], 
                      text_hash_to_text: Dict[str, str],
                      text_hash_to_prompt: Dict[str, str],
                      config: Dict[str, Any]) -> None:
    """Upload embeddings to Pinecone vector database."""
    
    if not HAS_PINECONE:
        print("Pinecone library not available. Skipping Pinecone upload.")
        return
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        # Get or create index
        index_name = config['pinecone']['index_name']
        
        # Check if index exists
        indexes = pc.list_indexes()
        index_names = [idx['name'] for idx in indexes]
        
        if index_name not in index_names:
            print(f"Index {index_name} not found. Creating new index...")
            
            # Create index
            pc.create_index(
                name=index_name,
                dimension=config['pinecone']['dimensions'],
                metric=config['pinecone']['metric'],
                spec={
                    "serverless": {
                        "cloud": config['pinecone']['cloud'],
                        "region": config['pinecone']['region']
                    }
                }
            )
            
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            time.sleep(10)
        
        # Get index
        index = pc.Index(index_name)
        
        # Prepare vectors for upload
        vectors = []
        for text_hash, embedding in embeddings.items():
            vector = {
                'id': text_hash,
                'values': embedding.tolist(),
                'metadata': {
                    'text': text_hash_to_text.get(text_hash, ''),
                    'prompt': text_hash_to_prompt.get(text_hash, ''),
                    'text_hash': text_hash
                }
            }
            vectors.append(vector)
        
        # Upload in batches
        batch_size = 100
        with tqdm(total=len(vectors), desc="Uploading to Pinecone") as pbar:
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                try:
                    index.upsert(batch)
                    pbar.update(len(batch))
                    time.sleep(0.1)  # Small delay to avoid rate limiting
                except Exception as e:
                    print(f"Error uploading batch {i//batch_size + 1}: {e}")
                    pbar.update(len(batch))
        
        print(f"Successfully uploaded {len(vectors)} vectors to Pinecone index {index_name}")
        
    except Exception as e:
        print(f"Error uploading to Pinecone: {e}")


def validate_embeddings(embeddings: Dict[str, np.ndarray], 
                       expected_dim: int) -> Tuple[bool, str]:
    """Validate that all embeddings have the correct dimensions."""
    
    if not embeddings:
        return False, "No embeddings found"
    
    for text_hash, embedding in embeddings.items():
        if len(embedding) != expected_dim:
            return False, f"Embedding {text_hash} has dimension {len(embedding)}, expected {expected_dim}"
    
    return True, f"All {len(embeddings)} embeddings have correct dimension {expected_dim}"


def main():
    """Main function to run the embedding generation process."""
    
    parser = argparse.ArgumentParser(description="Generate embeddings for dataset outputs")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for embedding generation")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--upload-pinecone", action="store_true", help="Upload embeddings to Pinecone")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regenerate all embeddings")
    
    args = parser.parse_args()
    
    # Load configuration and environment
    config = load_config(args.config)
    load_env_variables()
    
    # Validate API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    # Load dataset
    dataset_path = config['data']['dataset_path']
    df = pd.read_csv(dataset_path)
    
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Processing first {args.max_samples} samples")
    
    print(f"Processing {len(df)} samples from {dataset_path}")
    
    # Set up cache file
    cache_dir = Path(config['data']['embeddings_cache_path'])
    cache_format = config['data']['embeddings_format']
    cache_file = cache_dir / f"embeddings_cache.{cache_format}"
    
    # Force regenerate if requested
    if args.force_regenerate and cache_file.exists():
        cache_file.unlink()
        print("Existing cache file removed")
    
    # Generate embeddings
    embeddings = generate_embeddings(df, config, str(cache_file), args.batch_size)
    
    # Validate embeddings
    expected_dim = config['openai']['embedding_dimensions']
    is_valid, message = validate_embeddings(embeddings, expected_dim)
    print(f"Validation: {message}")
    
    if not is_valid:
        print("ERROR: Embedding validation failed")
        return
    
    # Create training data
    training_file = cache_dir / f"training_data.{cache_format}"
    create_training_data(df, embeddings, str(training_file))
    
    # Upload to Pinecone if requested
    if args.upload_pinecone:
        text_hash_to_text = {create_text_hash(str(row['output'])): str(row['output']) 
                            for _, row in df.iterrows()}
        text_hash_to_prompt = {create_text_hash(str(row['output'])): str(row['prompt']) 
                              for _, row in df.iterrows()}
        
        upload_to_pinecone(embeddings, text_hash_to_text, text_hash_to_prompt, config)
    
    # Print summary
    print("\n" + "="*60)
    print("EMBEDDING GENERATION SUMMARY")
    print("="*60)
    print(f"Total samples processed: {len(df):,}")
    print(f"Total embeddings generated: {len(embeddings):,}")
    print(f"Embedding dimensions: {expected_dim}")
    print(f"Cache file: {cache_file}")
    print(f"Training data file: {training_file}")
    
    if args.upload_pinecone:
        print(f"Pinecone index: {config['pinecone']['index_name']}")
    
    # Calculate estimated cost
    total_tokens = df['output'].str.len().sum()
    estimated_cost = (total_tokens / 1_000_000) * 0.02  # $0.02 per 1M tokens
    print(f"Estimated cost: ${estimated_cost:.4f}")


if __name__ == "__main__":
    main() 