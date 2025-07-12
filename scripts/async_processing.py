#!/usr/bin/env python3
"""
Async Processing Module for Embedding-to-Prompt Generation
Handles parallel processing with OpenAI rate limiting and efficient batch operations.
"""

import asyncio
import aiohttp
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
from datetime import datetime, timedelta
import os
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitTier(Enum):
    """OpenAI API rate limit tiers."""
    FREE = "free"       # 100 RPM, 2,000 RPD, 40,000 TPM
    TIER_1 = "tier_1"   # 3,000 RPM, 1,000,000 TPM, 3,000,000 batch queue
    TIER_2 = "tier_2"   # 5,000 RPM, 1,000,000 TPM, 20,000,000 batch queue  
    TIER_3 = "tier_3"   # 5,000 RPM, 5,000,000 TPM, 100,000,000 batch queue
    TIER_4 = "tier_4"   # 10,000 RPM, 5,000,000 TPM, 500,000,000 batch queue
    TIER_5 = "tier_5"   # 10,000 RPM, 10,000,000 TPM, 4,000,000,000 batch queue


@dataclass
class RateLimitConfig:
    """Configuration for OpenAI API rate limiting."""
    requests_per_minute: int
    requests_per_day: Optional[int]
    tokens_per_minute: int
    tokens_per_day: Optional[int]
    concurrent_requests: int
    batch_queue_limit: Optional[int]
    
    @classmethod
    def from_tier(cls, tier: RateLimitTier) -> "RateLimitConfig":
        """Create rate limit config from OpenAI tier."""
        configs = {
            RateLimitTier.FREE: cls(
                requests_per_minute=100,
                requests_per_day=2000,
                tokens_per_minute=40000,
                tokens_per_day=None,
                concurrent_requests=5,
                batch_queue_limit=None
            ),
            RateLimitTier.TIER_1: cls(
                requests_per_minute=3000,
                requests_per_day=None,
                tokens_per_minute=1000000,
                tokens_per_day=None,
                concurrent_requests=20,
                batch_queue_limit=3000000
            ),
            RateLimitTier.TIER_2: cls(
                requests_per_minute=5000,
                requests_per_day=None,
                tokens_per_minute=1000000,
                tokens_per_day=None,
                concurrent_requests=30,
                batch_queue_limit=20000000
            ),
            RateLimitTier.TIER_3: cls(
                requests_per_minute=5000,
                requests_per_day=None,
                tokens_per_minute=5000000,
                tokens_per_day=None,
                concurrent_requests=50,
                batch_queue_limit=100000000
            ),
            RateLimitTier.TIER_4: cls(
                requests_per_minute=10000,
                requests_per_day=None,
                tokens_per_minute=5000000,
                tokens_per_day=None,
                concurrent_requests=100,
                batch_queue_limit=500000000
            ),
            RateLimitTier.TIER_5: cls(
                requests_per_minute=10000,
                requests_per_day=None,
                tokens_per_minute=10000000,
                tokens_per_day=None,
                concurrent_requests=200,
                batch_queue_limit=4000000000
            ),
        }
        return configs[tier]


class OpenAIRateLimiter:
    """Advanced rate limiter for OpenAI API calls."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times: List[float] = []
        self.token_counts: List[Tuple[float, int]] = []
        self.semaphore = asyncio.Semaphore(config.concurrent_requests)
        self.lock = asyncio.Lock()
        
    async def acquire(self, estimated_tokens: int = 1000) -> None:
        """Acquire permission to make a request."""
        async with self.semaphore:
            await self._wait_for_rate_limit(estimated_tokens)
            
    async def _wait_for_rate_limit(self, estimated_tokens: int) -> None:
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        
        async with self.lock:
            # Clean old entries
            minute_ago = current_time - 60
            day_ago = current_time - 86400
            
            self.request_times = [t for t in self.request_times if t > minute_ago]
            self.token_counts = [(t, c) for t, c in self.token_counts if t > day_ago]
            
            # Check request rate limits
            if len(self.request_times) >= self.config.requests_per_minute:
                sleep_time = 60 - (current_time - self.request_times[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    await asyncio.sleep(sleep_time)
                    
            # Check token rate limits
            daily_tokens = sum(c for t, c in self.token_counts if t > day_ago)
            if daily_tokens + estimated_tokens > self.config.tokens_per_day:
                sleep_time = 86400 - (current_time - min(t for t, c in self.token_counts))
                if sleep_time > 0:
                    logger.warning(f"Daily token limit reached, sleeping for {sleep_time:.2f} seconds")
                    await asyncio.sleep(sleep_time)
                    
            # Record this request
            self.request_times.append(current_time)
            self.token_counts.append((current_time, estimated_tokens))


class AsyncEmbeddingProcessor:
    """Async processor for embedding-to-prompt generation."""
    
    def __init__(self, config: Dict[str, Any], rate_limit_tier: RateLimitTier = RateLimitTier.TIER_2):
        self.config = config
        self.rate_limit_config = RateLimitConfig.from_tier(rate_limit_tier)
        self.rate_limiter = OpenAIRateLimiter(self.rate_limit_config)
        
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            timeout=30.0,
            max_retries=3
        )
        
        logger.info(f"Initialized async processor with {rate_limit_tier.value} rate limits")
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def generate_embedding_async(self, text: str) -> np.ndarray:
        """Generate embedding for text with retry logic."""
        await self.rate_limiter.acquire(estimated_tokens=len(text.split()) * 2)
        
        try:
            response = await self.openai_client.embeddings.create(
                input=text,
                model=self.config['openai']['embedding_model']
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def generate_text_async(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """Generate text from prompt with retry logic."""
        await self.rate_limiter.acquire(estimated_tokens=len(prompt.split()) * 2 + 500)
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
            
    async def evaluate_prompt_async(self, prompt: str, target_embedding: np.ndarray) -> Dict[str, Any]:
        """Evaluate a prompt by generating text and comparing embeddings."""
        try:
            # Generate text from prompt
            generated_text = await self.generate_text_async(prompt)
            
            # Generate embedding for the generated text
            generated_embedding = await self.generate_embedding_async(generated_text)
            
            # Compute similarity
            similarity = np.dot(target_embedding, generated_embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(generated_embedding)
            )
            
            return {
                "similarity": float(similarity),
                "generated_text": generated_text,
                "prompt_length": len(prompt.split()),
                "response_length": len(generated_text.split()),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error evaluating prompt: {e}")
            return {
                "similarity": 0.0,
                "error": str(e),
                "success": False
            }
            
    async def process_batch_parallel(self, 
                                   embeddings: List[np.ndarray],
                                   prompts: List[str],
                                   max_concurrent: int = None) -> List[Dict[str, Any]]:
        """Process multiple embedding-prompt pairs in parallel."""
        if max_concurrent is None:
            max_concurrent = min(len(embeddings), self.rate_limit_config.concurrent_requests)
            
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(embedding: np.ndarray, prompt: str, index: int) -> Dict[str, Any]:
            async with semaphore:
                start_time = time.time()
                result = await self.evaluate_prompt_async(prompt, embedding)
                processing_time = time.time() - start_time
                
                return {
                    "index": index,
                    "processing_time": processing_time,
                    **result
                }
                
        # Create tasks for all embeddings
        tasks = [
            process_single(embedding, prompt, i)
            for i, (embedding, prompt) in enumerate(zip(embeddings, prompts))
        ]
        
        # Process all tasks concurrently
        logger.info(f"Processing {len(tasks)} items with {max_concurrent} concurrent workers")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
                processed_results.append({
                    "index": i,
                    "error": str(result),
                    "success": False
                })
            else:
                processed_results.append(result)
                
        return processed_results
        
    async def generate_diverse_prompts_async(self,
                                           target_embedding: np.ndarray,
                                           base_prompt: str,
                                           num_variants: int = 5) -> List[Dict[str, Any]]:
        """Generate diverse prompt variants and evaluate them."""
        
        # Generate variants with different temperatures
        variant_prompts = []
        for i in range(num_variants):
            temperature = 0.5 + (i * 0.3)  # Range from 0.5 to 1.7
            
            modification_prompt = f"""
            Rewrite this prompt to be more effective while maintaining the same intent.
            Make it {['more specific', 'more creative', 'more detailed', 'more concise', 'more engaging'][i % 5]}.
            
            Original prompt: {base_prompt}
            
            Rewritten prompt:
            """
            
            try:
                await self.rate_limiter.acquire(estimated_tokens=len(modification_prompt.split()) * 2)
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": modification_prompt}],
                    max_tokens=200,
                    temperature=temperature
                )
                
                variant_prompt = response.choices[0].message.content.strip()
                variant_prompts.append(variant_prompt)
                
            except Exception as e:
                logger.error(f"Error generating variant {i}: {e}")
                variant_prompts.append(base_prompt)  # Fallback to original
                
        # Evaluate all variants in parallel
        embeddings = [target_embedding] * len(variant_prompts)
        results = await self.process_batch_parallel(embeddings, variant_prompts)
        
        # Sort by similarity
        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return results
        
    async def close(self):
        """Close the async client."""
        await self.openai_client.close()


class ParallelPromptGenerator:
    """High-level interface for parallel prompt generation."""
    
    def __init__(self, 
                 inference_engine,
                 config: Dict[str, Any],
                 rate_limit_tier: RateLimitTier = RateLimitTier.TIER_2):
        self.inference_engine = inference_engine
        self.config = config
        self.async_processor = AsyncEmbeddingProcessor(config, rate_limit_tier)
        
    async def generate_prompts_with_evaluation(self,
                                             embeddings: List[Union[List[float], np.ndarray]],
                                             max_concurrent: int = 10) -> List[Dict[str, Any]]:
        """Generate prompts from embeddings and evaluate their quality."""
        
        # Convert embeddings to numpy arrays
        np_embeddings = []
        for emb in embeddings:
            if isinstance(emb, list):
                np_embeddings.append(np.array(emb))
            else:
                np_embeddings.append(emb)
                
        # Generate initial prompts using the model
        initial_prompts = []
        for embedding in np_embeddings:
            prompt = self.inference_engine.generate_prompt(embedding)
            initial_prompts.append(prompt)
            
        logger.info(f"Generated {len(initial_prompts)} initial prompts")
        
        # Evaluate prompts in parallel
        results = await self.async_processor.process_batch_parallel(
            np_embeddings, initial_prompts, max_concurrent
        )
        
        # Generate refined variants for the best prompts
        refined_results = []
        for i, result in enumerate(results):
            if result.get('success', False) and result.get('similarity', 0) > 0.7:
                # Generate variants for high-quality prompts
                variants = await self.async_processor.generate_diverse_prompts_async(
                    np_embeddings[i], initial_prompts[i], num_variants=3
                )
                
                refined_results.append({
                    "embedding_index": i,
                    "original_prompt": initial_prompts[i],
                    "original_result": result,
                    "variants": variants,
                    "best_variant": variants[0] if variants else None
                })
            else:
                refined_results.append({
                    "embedding_index": i,
                    "original_prompt": initial_prompts[i],
                    "original_result": result,
                    "variants": [],
                    "best_variant": None
                })
                
        return refined_results
        
    async def close(self):
        """Close resources."""
        await self.async_processor.close()


async def main():
    """Example usage of the parallel processing system."""
    
    # Load configuration
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Example embeddings (you would load these from your data)
    example_embeddings = [
        np.random.random(1024) for _ in range(10)
    ]
    
    # Initialize the processor
    from scripts.inference import EmbeddingToPromptInference
    inference_engine = EmbeddingToPromptInference("models/checkpoints/best_model.pt")
    
    generator = ParallelPromptGenerator(inference_engine, config)
    
    try:
        # Process embeddings in parallel
        start_time = time.time()
        results = await generator.generate_prompts_with_evaluation(
            example_embeddings, max_concurrent=5
        )
        processing_time = time.time() - start_time
        
        print(f"Processed {len(results)} embeddings in {processing_time:.2f} seconds")
        
        # Display results
        for i, result in enumerate(results):
            print(f"\nEmbedding {i + 1}:")
            print(f"  Original prompt: {result['original_prompt'][:100]}...")
            print(f"  Original similarity: {result['original_result'].get('similarity', 0):.3f}")
            
            if result['best_variant']:
                print(f"  Best variant similarity: {result['best_variant'].get('similarity', 0):.3f}")
                
    finally:
        await generator.close()


if __name__ == "__main__":
    asyncio.run(main()) 