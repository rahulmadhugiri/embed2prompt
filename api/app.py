#!/usr/bin/env python3
import os
import json
import yaml
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timedelta
import traceback

import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import numpy as np

import sys
sys.path.append('.')
from scripts.inference import EmbeddingToPromptInference
from scripts.async_processing import ParallelPromptGenerator, RateLimitTier
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

inference_engine = None
config = None
request_count = 0
start_time = time.time()

limiter = Limiter(key_func=get_remote_address)
security = HTTPBearer(auto_error=False)


class GeneratePromptRequest(BaseModel):
    target_embedding: List[float] = Field(..., description="Target embedding vector")
    return_metadata: bool = Field(default=False, description="Include generation metadata")
    generation_params: Optional[Dict[str, Any]] = Field(default=None, description="Generation parameters")
    
    @validator('target_embedding')
    def validate_embedding_length(cls, v):
        if len(v) != 1024:  # TODO: make configurable
            raise ValueError(f"Embedding must be exactly 1024 dimensions, got {len(v)}")
        return v


class GenerateFromTextRequest(BaseModel):
    text: str = Field(..., description="Input text to embed and generate prompt for")
    return_metadata: bool = Field(default=False)
    generation_params: Optional[Dict[str, Any]] = Field(default=None)
    
    @validator('text')
    def validate_text_length(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Text cannot be empty")
        if len(v) > 8192:  # OpenAI limit
            raise ValueError("Text too long (max 8192 characters)")
        return v


class BatchGenerateRequest(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    return_metadata: bool = Field(default=False)
    generation_params: Optional[Dict[str, Any]] = Field(default=None)
    
    @validator('embeddings')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError("Embeddings list cannot be empty")
        if len(v) > 100:  # Reasonable batch limit
            raise ValueError("Batch size too large (max 100)")
        return v


class ParallelGenerateRequest(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    max_concurrent: int = Field(default=10, ge=1, le=50, description="Max concurrent requests")
    rate_limit_tier: str = Field(default="tier_2", description="OpenAI rate limit tier")
    evaluate_quality: bool = Field(default=True, description="Evaluate prompt quality")
    generate_variants: bool = Field(default=False, description="Generate prompt variants")
    return_metadata: bool = Field(default=True, description="Include processing metadata")


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    if not credentials:
        return None
    expected_key = os.getenv('API_KEY')
    if expected_key and credentials.credentials != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine, config
    
    logger.info("Starting embedding-to-prompt API server...")
    
    try:
        config_path = os.getenv('CONFIG_PATH', 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_path = os.getenv('MODEL_PATH', 'models/checkpoints/best_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        inference_engine = EmbeddingToPromptInference(model_path, config_path)
        logger.info("Inference engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise
    
    yield
    
    logger.info("Shutting down embedding-to-prompt API server...")


app = FastAPI(
    title="Embedding-to-Prompt Generation API",
    description="Generate optimized prompts from target embeddings",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(SlowAPIMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('api', {}).get('cors_origins', ["*"]) if config else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.getenv('TRUSTED_HOSTS'):
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=os.getenv('TRUSTED_HOSTS').split(',')
    )


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    global request_count
    start_time = time.time()
    request_count += 1
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response


@app.get("/", response_class=HTMLResponse)
async def root():
    uptime = time.time() - start_time
    return f"""
    <html><body>
    <h1>Embedding-to-Prompt Generation API</h1>
    <p>Status: Running</p>
    <p>Uptime: {uptime:.1f} seconds</p>
    <p>Requests served: {request_count}</p>
    <p><a href="/docs">API Documentation</a></p>
    </body></html>
    """


@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "healthy",
        "uptime": time.time() - start_time,
        "requests_served": request_count,
        "model_architecture": config['model']['architecture'] if config else "unknown"
    }


@app.post("/generate_prompt")
@limiter.limit("30/minute")
async def generate_prompt(
    request: Request,
    req: GeneratePromptRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
    try:
        start_time = time.time()
        
        embedding = np.array(req.target_embedding)
        prompt = inference_engine.generate_prompt(
            embedding, 
            generation_params=req.generation_params or {}
        )
        
        processing_time = time.time() - start_time
        
        result = {
            "prompt": prompt,
            "success": True
        }
        
        if req.return_metadata:
            result["metadata"] = {
                "processing_time": processing_time,
                "embedding_dimension": len(req.target_embedding),
                "model_architecture": config['model']['architecture']
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate_prompt_from_text")
@limiter.limit("20/minute")
async def generate_prompt_from_text(
    request: Request,
    req: GenerateFromTextRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
    try:
        start_time = time.time()
        
        result = inference_engine.generate_prompt_from_text(
            req.text,
            return_metadata=req.return_metadata,
            generation_params=req.generation_params or {}
        )
        
        result["processing_time"] = time.time() - start_time
        return result
        
    except Exception as e:
        logger.error(f"Error generating prompt from text: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate_prompts_batch")
@limiter.limit("10/minute")
async def generate_prompts_batch(
    request: Request,
    req: BatchGenerateRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
    try:
        start_time = time.time()
        
        embeddings = [np.array(emb) for emb in req.embeddings]
        results = []
        
        for i, embedding in enumerate(embeddings):
            try:
                prompt = inference_engine.generate_prompt(
                    embedding,
                    generation_params=req.generation_params or {}
                )
                results.append({
                    "index": i,
                    "prompt": prompt,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "prompt": None,
                    "success": False,
                    "error": str(e)
                })
        
        response = {
            "results": results,
            "total_processed": len(embeddings),
            "successful": sum(1 for r in results if r["success"]),
            "processing_time": time.time() - start_time
        }
        
        if req.return_metadata:
            response["metadata"] = {
                "batch_size": len(embeddings),
                "model_architecture": config['model']['architecture']
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@app.post("/generate_prompts_parallel")
@limiter.limit("5/minute")
async def generate_prompts_parallel(
    request: Request,
    req: ParallelGenerateRequest,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(verify_api_key)
):
    try:
        rate_tier = RateLimitTier(req.rate_limit_tier)
        generator = ParallelPromptGenerator(inference_engine, config, rate_tier)
        
        start_time = time.time()
        results = await generator.generate_prompts_with_evaluation(
            req.embeddings,
            max_concurrent=req.max_concurrent
        )
        processing_time = time.time() - start_time
        
        # Cleanup in background
        background_tasks.add_task(generator.close)
        
        successful = sum(1 for r in results if r.get('original_result', {}).get('success', False))
        
        response = {
            "results": results,
            "summary": {
                "total_processed": len(req.embeddings),
                "successful": successful,
                "failed": len(req.embeddings) - successful,
                "success_rate": successful / len(req.embeddings) if req.embeddings else 0,
                "processing_time": processing_time
            }
        }
        
        if req.return_metadata:
            avg_similarity = np.mean([
                r.get('original_result', {}).get('similarity', 0) 
                for r in results if r.get('original_result', {}).get('success', False)
            ]) if successful > 0 else 0
            
            response["summary"].update({
                "average_similarity": float(avg_similarity),
                "max_similarity": float(max([
                    r.get('original_result', {}).get('similarity', 0) 
                    for r in results if r.get('original_result', {}).get('success', False)
                ], default=[0])),
                "min_similarity": float(min([
                    r.get('original_result', {}).get('similarity', 0) 
                    for r in results if r.get('original_result', {}).get('success', False)
                ], default=[0]))
            })
            
            response["processing_details"] = {
                "max_concurrent": req.max_concurrent,
                "rate_limiting": rate_tier.value,
                "quality_evaluation": req.evaluate_quality,
                "variant_generation": req.generate_variants
            }
            
            response["performance_metrics"] = {
                "requests_per_second": len(req.embeddings) / processing_time if processing_time > 0 else 0,
                "average_processing_time": processing_time / len(req.embeddings) if req.embeddings else 0
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        raise HTTPException(status_code=500, detail=f"Parallel processing failed: {str(e)}")


if __name__ == "__main__":
    host = config.get('api', {}).get('host', '0.0.0.0') if config else '0.0.0.0'
    port = config.get('api', {}).get('port', 8000) if config else 8000
    debug = config.get('api', {}).get('debug', False) if config else False
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    ) 