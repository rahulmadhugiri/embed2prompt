#!/usr/bin/env python3
"""
Inference Script for Embedding-to-Prompt Model
Generates prompts from target embeddings using trained models.
"""

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

# Import our model architecture
import sys
sys.path.append('.')
from models.architecture import EmbeddingToPromptModel
from dotenv import load_dotenv

# Optional imports for evaluation
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingToPromptInference:
    """Class for performing inference with embedding-to-prompt models."""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model
            config_path: Path to the configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load environment variables
        load_dotenv()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Initialize OpenAI client if available
        self.openai_client = None
        if HAS_OPENAI and os.getenv('OPENAI_API_KEY'):
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        logger.info("Inference engine initialized successfully")
    
    def _load_model(self) -> EmbeddingToPromptModel:
        """Load the trained model."""
        try:
            # Try loading from our custom format first
            model = EmbeddingToPromptModel.load_pretrained(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load from custom format: {e}")
            
            # Try loading from checkpoint
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
        """
        Generate a single prompt from an embedding.
        
        Args:
            embedding: Input embedding vector
            generation_params: Optional generation parameters
            
        Returns:
            Generated prompt string
        """
        if generation_params is None:
            generation_params = {}
        
        # Convert to tensor
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Generate prompt
        with torch.no_grad():
            prompts = self.model.generate_prompt(embedding_tensor, **generation_params)
        
        return prompts[0] if prompts else ""
    
    def generate_prompts_batch(self, embeddings: List[Union[np.ndarray, List[float]]], 
                              generation_params: Optional[Dict[str, Any]] = None,
                              batch_size: int = 32) -> List[str]:
        """
        Generate prompts for a batch of embeddings.
        
        Args:
            embeddings: List of embedding vectors
            generation_params: Optional generation parameters
            batch_size: Batch size for processing
            
        Returns:
            List of generated prompt strings
        """
        if generation_params is None:
            generation_params = {}
        
        prompts = []
        
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Generating prompts"):
            batch_embeddings = embeddings[i:i + batch_size]
            
            # Convert to tensor
            batch_tensor = []
            for emb in batch_embeddings:
                if isinstance(emb, list):
                    emb = np.array(emb)
                batch_tensor.append(torch.tensor(emb, dtype=torch.float32))
            
            batch_tensor = torch.stack(batch_tensor).to(self.device)
            
            # Generate prompts
            with torch.no_grad():
                batch_prompts = self.model.generate_prompt(batch_tensor, **generation_params)
            
            prompts.extend(batch_prompts)
        
        return prompts
    
    def generate_prompt_from_text(self, text: str, 
                                 generation_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a prompt from text by first embedding it.
        
        Args:
            text: Input text to embed
            generation_params: Optional generation parameters
            
        Returns:
            Generated prompt string
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not available. Please set OPENAI_API_KEY.")
        
        # Generate embedding
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.config['openai']['embedding_model']
        )
        
        embedding = response.data[0].embedding
        
        # Generate prompt
        return self.generate_prompt(embedding, generation_params)
    
    def generate_similar_prompts(self, target_embedding: Union[np.ndarray, List[float]],
                                num_candidates: int = 5,
                                diversity_penalty: float = 0.1) -> List[str]:
        """
        Generate multiple diverse prompts for the same embedding.
        
        Args:
            target_embedding: Target embedding vector
            num_candidates: Number of prompts to generate
            diversity_penalty: Penalty for similar prompts
            
        Returns:
            List of diverse prompt strings
        """
        prompts = []
        
        for i in range(num_candidates):
            # Use different generation parameters for diversity
            generation_params = {
                'temperature': 0.7 + (i * 0.1),
                'top_p': 0.9 - (i * 0.05),
                'do_sample': True
            }
            
            prompt = self.generate_prompt(target_embedding, generation_params)
            
            # Simple diversity check
            if not any(self._compute_similarity(prompt, existing) > (1 - diversity_penalty) 
                      for existing in prompts):
                prompts.append(prompt)
        
        return prompts
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute simple similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_prompt_quality(self, prompt: str, target_embedding: Union[np.ndarray, List[float]],
                               test_model: str = "gpt-3.5-turbo") -> Dict[str, float]:
        """
        Evaluate the quality of a generated prompt by testing it with an LLM.
        
        Args:
            prompt: Generated prompt
            target_embedding: Target embedding
            test_model: Model to use for testing
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.openai_client:
            logger.warning("OpenAI client not available. Skipping evaluation.")
            return {}
        
        try:
            # Generate response using the prompt
            response = self.openai_client.chat.completions.create(
                model=test_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content
            
            # Embed the generated text
            embedding_response = self.openai_client.embeddings.create(
                input=generated_text,
                model=self.config['openai']['embedding_model']
            )
            
            generated_embedding = np.array(embedding_response.data[0].embedding)
            
            # Convert target embedding to numpy array
            if isinstance(target_embedding, list):
                target_embedding = np.array(target_embedding)
            
            # Compute cosine similarity
            cosine_similarity = np.dot(target_embedding, generated_embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(generated_embedding)
            )
            
            return {
                'cosine_similarity': float(cosine_similarity),
                'generated_text': generated_text,
                'prompt_length': len(prompt.split()),
                'response_length': len(generated_text.split())
            }
            
        except Exception as e:
            logger.error(f"Error evaluating prompt: {e}")
            return {}
    
    def search_similar_embeddings(self, target_embedding: Union[np.ndarray, List[float]],
                                 top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings in the vector database.
        
        Args:
            target_embedding: Target embedding vector
            top_k: Number of similar embeddings to return
            
        Returns:
            List of similar embeddings with metadata
        """
        if not HAS_PINECONE:
            logger.warning("Pinecone not available. Cannot search similar embeddings.")
            return []
        
        try:
            # Initialize Pinecone
            pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index = pc.Index(self.config['pinecone']['index_name'])
            
            # Convert embedding to list
            if isinstance(target_embedding, np.ndarray):
                target_embedding = target_embedding.tolist()
            
            # Search for similar vectors
            results = index.query(
                vector=target_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return results.matches
            
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            return []
    
    def batch_process_file(self, input_file: str, output_file: str, 
                          embedding_column: str = 'embedding',
                          generation_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Process a file of embeddings and generate prompts.
        
        Args:
            input_file: Path to input file (JSON/JSONL)
            output_file: Path to output file
            embedding_column: Column name containing embeddings
            generation_params: Optional generation parameters
        """
        logger.info(f"Processing file: {input_file}")
        
        # Load data
        if input_file.endswith('.jsonl'):
            import jsonlines
            with jsonlines.open(input_file, 'r') as reader:
                data = [item for item in reader]
        else:
            with open(input_file, 'r') as f:
                data = json.load(f)
        
        # Extract embeddings
        embeddings = [item[embedding_column] for item in data]
        
        # Generate prompts
        prompts = self.generate_prompts_batch(embeddings, generation_params)
        
        # Add prompts to data
        for i, prompt in enumerate(prompts):
            data[i]['generated_prompt'] = prompt
        
        # Save results
        if output_file.endswith('.jsonl'):
            import jsonlines
            with jsonlines.open(output_file, 'w') as writer:
                for item in data:
                    writer.write(item)
        else:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")


def main():
    """Main function for command-line interface."""
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
    
    # Initialize inference engine
    inference = EmbeddingToPromptInference(args.model_path, args.config)
    
    # Generation parameters
    generation_params = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'do_sample': True
    }
    
    if args.embedding:
        # Single embedding inference
        try:
            embedding = json.loads(args.embedding)
            
            if args.num_candidates == 1:
                prompt = inference.generate_prompt(embedding, generation_params)
                print(f"Generated prompt: {prompt}")
            else:
                prompts = inference.generate_similar_prompts(embedding, args.num_candidates)
                for i, prompt in enumerate(prompts, 1):
                    print(f"Prompt {i}: {prompt}")
            
            # Evaluate if requested
            if args.evaluate:
                evaluation = inference.evaluate_prompt_quality(prompt, embedding)
                print(f"Evaluation metrics: {evaluation}")
                
        except Exception as e:
            logger.error(f"Error processing embedding: {e}")
    
    elif args.text:
        # Text-to-prompt inference
        try:
            prompt = inference.generate_prompt_from_text(args.text, generation_params)
            print(f"Generated prompt: {prompt}")
            
            # Evaluate if requested
            if args.evaluate:
                # Need to get embedding first
                embedding = inference.openai_client.embeddings.create(
                    input=args.text,
                    model=inference.config['openai']['embedding_model']
                ).data[0].embedding
                
                evaluation = inference.evaluate_prompt_quality(prompt, embedding)
                print(f"Evaluation metrics: {evaluation}")
                
        except Exception as e:
            logger.error(f"Error processing text: {e}")
    
    elif args.file:
        # Batch file processing
        if not args.output:
            logger.error("Output file required for batch processing")
            return
        
        try:
            inference.batch_process_file(
                args.file, 
                args.output, 
                args.embedding_column,
                generation_params
            )
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
    
    else:
        # Interactive mode
        print("Interactive mode. Enter embeddings as JSON arrays or 'quit' to exit.")
        
        while True:
            try:
                user_input = input("Enter embedding: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                embedding = json.loads(user_input)
                prompt = inference.generate_prompt(embedding, generation_params)
                print(f"Generated prompt: {prompt}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main() 