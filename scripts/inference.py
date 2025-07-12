#!/usr/bin/env python3
import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
from tqdm import tqdm
import time

import sys
sys.path.append('.')
from models.architecture import EmbeddingToPromptModel
from dotenv import load_dotenv

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import pinecone
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingToPromptInference:
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        self.model_path = model_path
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        load_dotenv()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = self._load_model()
        self.model.eval()
        
        self.openai_client = None
        if HAS_OPENAI and os.getenv('OPENAI_API_KEY'):
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        logger.info("Inference engine initialized successfully")
    
    def _load_model(self) -> EmbeddingToPromptModel:
        try:
            model = EmbeddingToPromptModel.load_pretrained(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load from custom format: {e}")
            try:
                checkpoint = torch.load(os.path.join(self.model_path, 'model.pt'))
                model = EmbeddingToPromptModel(checkpoint['config'])
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from checkpoint")
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                raise
        
        model.to(self.device)
        return model
    
    def generate_prompt(self, embedding: Union[np.ndarray, List[float]], 
                       generation_params: Optional[Dict[str, Any]] = None) -> str:
        if generation_params is None:
            generation_params = {}
        
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prompts = self.model.generate_prompt(embedding_tensor, **generation_params)
        
        return prompts[0] if prompts else ""
    
    def generate_prompts_batch(self, embeddings: List[Union[np.ndarray, List[float]]], 
                              generation_params: Optional[Dict[str, Any]] = None) -> List[str]:
        if generation_params is None:
            generation_params = {}
        
        # Convert to tensor batch
        embedding_arrays = []
        for emb in embeddings:
            if isinstance(emb, list):
                embedding_arrays.append(np.array(emb))
            else:
                embedding_arrays.append(emb)
        
        embedding_tensor = torch.tensor(np.stack(embedding_arrays), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            prompts = self.model.generate_prompt(embedding_tensor, **generation_params)
        
        return prompts
    
    def generate_prompt_from_text(self, text: str, return_metadata: bool = False,
                                 generation_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.openai_client:
            raise ValueError("OpenAI client not available. Please set OPENAI_API_KEY.")
        
        if generation_params is None:
            generation_params = {}
        
        start_time = time.time()
        
        try:
            # Generate embedding
            embedding_response = self.openai_client.embeddings.create(
                input=text,
                model=self.config['openai']['embedding_model']
            )
            embedding = np.array(embedding_response.data[0].embedding)
            
            # Generate prompt
            prompt = self.generate_prompt(embedding, generation_params)
            
            result = {
                "prompt": prompt,
                "success": True
            }
            
            if return_metadata:
                result["metadata"] = {
                    "text_length": len(text),
                    "embedding_dimension": len(embedding),
                    "generation_time": time.time() - start_time
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating prompt from text: {e}")
            return {
                "prompt": None,
                "success": False,
                "error": str(e)
            }
    
    def evaluate_prompt_quality(self, original_embedding: np.ndarray, 
                               generated_prompt: str) -> Dict[str, Any]:
        if not self.openai_client:
            logger.warning("OpenAI client not available for quality evaluation")
            return {"success": False, "error": "OpenAI client not available"}
        
        try:
            # Generate embedding for the prompt
            prompt_embedding_response = self.openai_client.embeddings.create(
                input=generated_prompt,
                model=self.config['openai']['embedding_model']
            )
            prompt_embedding = np.array(prompt_embedding_response.data[0].embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(original_embedding, prompt_embedding) / (
                np.linalg.norm(original_embedding) * np.linalg.norm(prompt_embedding)
            )
            
            return {
                "success": True,
                "similarity": float(similarity),
                "prompt_length": len(generated_prompt.split()),
                "original_embedding_norm": float(np.linalg.norm(original_embedding)),
                "prompt_embedding_norm": float(np.linalg.norm(prompt_embedding))
            }
            
        except Exception as e:
            logger.error(f"Error evaluating prompt quality: {e}")
            return {"success": False, "error": str(e)}
    
    def search_similar_embeddings(self, target_embedding: np.ndarray, 
                                 top_k: int = 5) -> Dict[str, Any]:
        if not HAS_PINECONE:
            return {"success": False, "error": "Pinecone not available"}
        
        try:
            # TODO: Implement Pinecone search
            logger.warning("Pinecone search not yet implemented")
            return {"success": False, "error": "Not implemented"}
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            return {"success": False, "error": str(e)}


def process_file(inference: EmbeddingToPromptInference, file_path: str, 
                embedding_column: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
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
        
        results = []
        for i, item in enumerate(tqdm(data, desc="Processing embeddings")):
            try:
                embedding = item[embedding_column]
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)
                
                prompt = inference.generate_prompt(embedding)
                
                result = {
                    "index": i,
                    "prompt": prompt,
                    "success": True
                }
                
                # Copy over any additional fields
                for key, value in item.items():
                    if key != embedding_column:
                        result[key] = value
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing item {i}: {e}")
                results.append({
                    "index": i,
                    "prompt": None,
                    "success": False,
                    "error": str(e)
                })
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Generate prompts from embeddings")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--embedding", help="Single embedding as JSON string")
    parser.add_argument("--text", help="Text to embed and generate prompt for")
    parser.add_argument("--file", help="File containing embeddings to process")
    parser.add_argument("--output", help="Output file for batch processing")
    parser.add_argument("--embedding-column", default="embedding", help="Column name for embeddings")
    parser.add_argument("--num-candidates", type=int, default=1, help="Number of prompt candidates")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate prompt quality")
    parser.add_argument("--search-similar", action="store_true", help="Search for similar embeddings")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    
    args = parser.parse_args()
    
    inference = EmbeddingToPromptInference(args.model_path, args.config)
    
    generation_params = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'do_sample': True
    }
    
    if args.embedding:
        # Single embedding
        embedding = np.array(json.loads(args.embedding))
        prompt = inference.generate_prompt(embedding, generation_params)
        print(f"Generated prompt: {prompt}")
        
        if args.evaluate:
            quality = inference.evaluate_prompt_quality(embedding, prompt)
            print(f"Quality evaluation: {quality}")
        
        if args.search_similar:
            similar = inference.search_similar_embeddings(embedding)
            print(f"Similar embeddings: {similar}")
    
    elif args.text:
        # Text to prompt
        result = inference.generate_prompt_from_text(
            args.text, 
            return_metadata=True,
            generation_params=generation_params
        )
        print(f"Generated prompt: {result['prompt']}")
        if result.get('metadata'):
            print(f"Metadata: {result['metadata']}")
    
    elif args.file:
        # Batch processing
        results = process_file(
            inference, 
            args.file, 
            args.embedding_column, 
            args.output
        )
        
        successful = sum(1 for r in results if r['success'])
        print(f"Processed {len(results)} items, {successful} successful")
    
    else:
        print("Please provide either --embedding, --text, or --file")


if __name__ == "__main__":
    main() 