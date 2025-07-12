#!/usr/bin/env python3
import os
import json
import yaml
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup consistent logging across the project."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load config from {config_path}: {e}")


def load_embeddings_from_file(file_path: str, embedding_key: str = "embedding") -> List[np.ndarray]:
    """Load embeddings from various file formats."""
    embeddings = []
    
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif file_path.endswith('.jsonl'):
            import jsonlines
            with jsonlines.open(file_path) as reader:
                data = list(reader)
        else:
            import pandas as pd
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
        
        for item in data:
            if embedding_key in item:
                embedding = item[embedding_key]
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)
                embeddings.append(np.array(embedding))
        
        return embeddings
        
    except Exception as e:
        raise ValueError(f"Failed to load embeddings from {file_path}: {e}")


def save_results_to_file(results: List[Dict[str, Any]], output_path: str):
    """Save results to file in appropriate format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.jsonl':
        import jsonlines
        with jsonlines.open(output_path, 'w') as writer:
            for result in results:
                writer.write(result)
    else:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0


def validate_embedding_dimensions(embeddings: List[np.ndarray], expected_dim: int = 1024) -> Dict[str, Any]:
    """Validate that embeddings have the expected dimensions."""
    if not embeddings:
        return {"valid": False, "error": "No embeddings provided"}
    
    invalid_indices = []
    for i, emb in enumerate(embeddings):
        if emb.shape[0] != expected_dim:
            invalid_indices.append(i)
    
    return {
        "valid": len(invalid_indices) == 0,
        "total": len(embeddings),
        "invalid_count": len(invalid_indices),
        "invalid_indices": invalid_indices,
        "expected_dim": expected_dim
    }


def create_test_embeddings(count: int, dimensions: int = 1024, seed: int = 42) -> List[np.ndarray]:
    """Create random test embeddings for testing purposes."""
    np.random.seed(seed)
    embeddings = []
    
    for i in range(count):
        # Create varied embeddings around different centers
        center = np.random.randn(dimensions) * 0.1
        noise = np.random.randn(dimensions) * 0.01
        embedding = center + noise
        embeddings.append(embedding)
    
    return embeddings


def get_model_path(config: Optional[Dict[str, Any]] = None) -> str:
    """Get the model path from environment or config."""
    # Try environment variable first
    model_path = os.getenv('MODEL_PATH')
    if model_path and os.path.exists(model_path):
        return model_path
    
    # Try config file
    if config and 'model' in config and 'checkpoint_path' in config['model']:
        model_path = config['model']['checkpoint_path']
        if os.path.exists(model_path):
            return model_path
    
    # Default path
    default_path = "models/checkpoints/best_model.pt"
    if os.path.exists(default_path):
        return default_path
    
    raise FileNotFoundError("Model not found. Please specify MODEL_PATH or ensure model exists at default location.")


def ensure_directory_exists(path: Union[str, Path]):
    """Ensure that a directory exists, creating it if necessary."""
    Path(path).mkdir(parents=True, exist_ok=True)


def format_processing_time(seconds: float) -> str:
    """Format processing time in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_embedding_stats(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """Calculate basic statistics for a list of embeddings."""
    if not embeddings:
        return {}
    
    all_values = np.concatenate(embeddings)
    
    return {
        "count": len(embeddings),
        "dimensions": embeddings[0].shape[0] if embeddings else 0,
        "mean": float(np.mean(all_values)),
        "std": float(np.std(all_values)),
        "min": float(np.min(all_values)),
        "max": float(np.max(all_values))
    }


def safe_json_serialize(obj: Any) -> Any:
    """Safely serialize objects to JSON, handling numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    else:
        return obj 