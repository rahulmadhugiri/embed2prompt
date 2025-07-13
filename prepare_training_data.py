#!/usr/bin/env python3
"""
Prepare Training Data from Pinecone
Fetches embeddings and prompts from Pinecone and formats them for training.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any
import jsonlines
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from pinecone import Pinecone
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False
    print("⚠️  Pinecone not available. Please install with: pip install pinecone-client")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def fetch_pinecone_data(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch all training data from Pinecone."""
    if not HAS_PINECONE:
        raise ImportError("Pinecone client not available")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(config['pinecone']['index_name'])
    
    # Fetch all vectors with metadata
    logger.info("Fetching all data from Pinecone...")
    
    # Get index stats to know how many vectors we have
    stats = index.describe_index_stats()
    total_vectors = stats['total_vector_count']
    logger.info(f"Total vectors in index: {total_vectors}")
    
    all_data = []
    
    # Simple approach: Try smaller batches and handle missing IDs gracefully
    logger.info("Fetching all vectors with small batches...")
    
    # Use much smaller batches to avoid 414 errors
    batch_size = 100  # Much smaller batches
    max_attempts = 52002
    
    for start_idx in range(0, max_attempts, batch_size):
        end_idx = min(start_idx + batch_size, max_attempts)
        batch_ids = [f"alpaca-{i}" for i in range(start_idx, end_idx)]
        
        try:
            fetch_response = index.fetch(ids=batch_ids)
            
            # Process only the vectors that actually exist
            found_vectors = fetch_response.get('vectors', {})
            
            for vector_id, vector_data in found_vectors.items():
                if vector_data and vector_data.get('metadata', {}).get('type') == 'alpaca':
                    all_data.append({
                        'id': vector_id,
                        'embedding': vector_data['values'],
                        'prompt': vector_data['metadata']['prompt'],
                        'output': vector_data['metadata']['output'],
                        'tokens': vector_data['metadata']['tokens']
                    })
            
            if found_vectors:
                logger.info(f"Batch {start_idx//batch_size + 1}: Found {len(found_vectors)} vectors, total so far: {len(all_data)}")
            
            # Stop if we've found enough vectors
            if len(all_data) >= total_vectors:
                logger.info(f"Reached target of {total_vectors} vectors, stopping.")
                break
                
        except Exception as batch_error:
            # Log but continue - this is expected for batches with mostly missing IDs
            if "414" not in str(batch_error):
                logger.warning(f"Error fetching batch {start_idx//batch_size + 1}: {batch_error}")
            continue
    
    logger.info(f"Successfully fetched {len(all_data)} vectors from Pinecone")
    return all_data


def format_for_training(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format data for training (embedding -> prompt pairs)."""
    training_data = []
    
    for item in data:
        training_item = {
            'embedding': item['embedding'],
            'prompt': item['prompt'],
            'output': item['output'],  # Keep for reference
            'tokens': item['tokens'],
            'id': item['id']
        }
        training_data.append(training_item)
    
    logger.info(f"Formatted {len(training_data)} training examples")
    return training_data


def save_training_data(data: List[Dict[str, Any]], output_path: str):
    """Save training data to JSONL format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(output_path, 'w') as writer:
        for item in data:
            writer.write(item)
    
    logger.info(f"Saved {len(data)} training examples to {output_path}")


def main():
    """Main function to prepare training data."""
    logger.info("Starting training data preparation...")
    
    # Load configuration
    config = load_config()
    
    # Create output directory
    output_dir = Path(config['data']['embeddings_cache_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data from Pinecone
    logger.info("Fetching data from Pinecone...")
    pinecone_data = fetch_pinecone_data(config)
    
    if not pinecone_data:
        logger.error("No data fetched from Pinecone!")
        return
    
    # Format for training
    logger.info("Formatting data for training...")
    training_data = format_for_training(pinecone_data)
    
    # Save training data
    output_path = output_dir / 'training_data.jsonl'
    save_training_data(training_data, output_path)
    
    # Print summary
    logger.info("=" * 50)
    logger.info("TRAINING DATA PREPARATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total examples: {len(training_data)}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Format: JSONL with 'embedding' and 'prompt' fields")
    
    # Show sample
    if training_data:
        sample = training_data[0]
        logger.info(f"\nSample data:")
        logger.info(f"  ID: {sample['id']}")
        logger.info(f"  Prompt: {sample['prompt'][:100]}...")
        logger.info(f"  Embedding dim: {len(sample['embedding'])}")
        logger.info(f"  Tokens: {sample['tokens']}")


if __name__ == "__main__":
    main() 