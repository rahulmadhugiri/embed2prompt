#!/usr/bin/env python3
"""
FastAPI Server for Embedding-to-Prompt Generation
Serves the trained embedding-to-prompt model via REST API.
"""

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

# Import our inference engine
import sys
sys.path.append('.')
from scripts.inference import EmbeddingToPromptInference
from scripts.async_processing import ParallelPromptGenerator, RateLimitTier
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variables
inference_engine = None
config = None
request_count = 0
start_time = time.time()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer(auto_error=False)


# Pydantic models
class EmbeddingRequest(BaseModel):
    """Request model for embedding-to-prompt generation."""
    
    target_embedding: List[float] = Field(
        ...,
        description="Target embedding vector (1024 dimensions for text-embedding-3-small)",
        example=[0.1] * 1024
    )
    
    generation_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional generation parameters",
        example={
            "temperature": 0.7,
            "top_p": 0.9,
            "num_candidates": 1,
            "max_length": 100
        }
    )
    
    return_metadata: Optional[bool] = Field(
        default=False,
        description="Whether to return additional metadata"
    )
    
    @validator('target_embedding')
    def validate_embedding_dimensions(cls, v):
        if len(v) != 1024:
            raise ValueError("Embedding must have exactly 1024 dimensions")
        return v


class TextRequest(BaseModel):
    """Request model for text-to-prompt generation."""
    
    text: str = Field(
        ...,
        description="Input text to embed and generate prompt for",
        example="How to build a web application"
    )
    
    generation_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional generation parameters"
    )
    
    return_metadata: Optional[bool] = Field(
        default=False,
        description="Whether to return additional metadata"
    )
    
    @validator('text')
    def validate_text_length(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Text cannot be empty")
        if len(v) > 8000:  # Reasonable limit for embedding
            raise ValueError("Text too long (max 8000 characters)")
        return v


class BatchRequest(BaseModel):
    """Request model for batch embedding-to-prompt generation."""
    
    embeddings: List[List[float]] = Field(
        ...,
        description="List of target embedding vectors",
        example=[[0.1] * 1024, [0.2] * 1024]
    )
    
    generation_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional generation parameters"
    )
    
    return_metadata: Optional[bool] = Field(
        default=False,
        description="Whether to return additional metadata"
    )
    
    @validator('embeddings')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError("Embeddings list cannot be empty")
        if len(v) > 50:  # Reasonable batch size limit
            raise ValueError("Batch size too large (max 50 embeddings)")
        
        for i, emb in enumerate(v):
            if len(emb) != 1024:
                raise ValueError(f"Embedding {i} must have exactly 1024 dimensions")
        
        return v


class PromptResponse(BaseModel):
    """Response model for prompt generation."""
    
    prompt: str = Field(
        ...,
        description="Generated prompt",
        example="Explain the process of building a modern web application"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the generation"
    )
    
    timestamp: str = Field(
        ...,
        description="Generation timestamp",
        example="2024-01-01T12:00:00Z"
    )
    
    request_id: str = Field(
        ...,
        description="Unique request identifier"
    )


class BatchResponse(BaseModel):
    """Response model for batch prompt generation."""
    
    prompts: List[str] = Field(
        ...,
        description="List of generated prompts"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the generation"
    )
    
    timestamp: str = Field(
        ...,
        description="Generation timestamp"
    )
    
    request_id: str = Field(
        ...,
        description="Unique request identifier"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    uptime: float = Field(..., description="Uptime in seconds")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    requests_processed: int = Field(..., description="Total requests processed")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    request_id: str = Field(..., description="Request identifier")
    timestamp: str = Field(..., description="Error timestamp")


class ParallelBatchRequest(BaseModel):
    """Request model for parallel batch processing with evaluation."""
    
    embeddings: List[List[float]] = Field(
        ...,
        description="List of target embedding vectors",
        example=[[0.1] * 1024, [0.2] * 1024]
    )
    
    max_concurrent: Optional[int] = Field(
        default=10,
        description="Maximum number of concurrent requests",
        ge=1,
        le=50
    )
    
    rate_limit_tier: Optional[str] = Field(
        default="tier_2",
        description="OpenAI rate limit tier",
        regex="^tier_[1-5]$"
    )
    
    evaluate_quality: Optional[bool] = Field(
        default=True,
        description="Whether to evaluate prompt quality"
    )
    
    generate_variants: Optional[bool] = Field(
        default=False,
        description="Whether to generate refined variants"
    )
    
    return_metadata: Optional[bool] = Field(
        default=True,
        description="Whether to return detailed metadata"
    )
    
    @validator('embeddings')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError("Embeddings list cannot be empty")
        if len(v) > 50:  # Reasonable batch size limit
            raise ValueError("Batch size too large (max 50 embeddings)")
        
        for i, emb in enumerate(v):
            if len(emb) != 1024:
                raise ValueError(f"Embedding {i} must have exactly 1024 dimensions")
        
        return v


class ParallelBatchResponse(BaseModel):
    """Response model for parallel batch processing."""
    
    results: List[Dict[str, Any]] = Field(
        ...,
        description="List of processing results with evaluation data"
    )
    
    summary: Dict[str, Any] = Field(
        ...,
        description="Summary statistics of the batch processing"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the processing"
    )
    
    timestamp: str = Field(
        ...,
        description="Processing timestamp"
    )
    
    request_id: str = Field(
        ...,
        description="Unique request identifier"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    global inference_engine, config
    
    # Startup
    logger.info("Starting embedding-to-prompt API server...")
    
    try:
        # Load configuration
        config_path = os.getenv('CONFIG_PATH', 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize inference engine
        model_path = os.getenv('MODEL_PATH', 'models/checkpoints/best_model.pt')
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        inference_engine = EmbeddingToPromptInference(model_path, config_path)
        logger.info("Inference engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down embedding-to-prompt API server...")


# Initialize FastAPI app
app = FastAPI(
    title="Embedding-to-Prompt Generation API",
    description="Generate optimized prompts from target embeddings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(SlowAPIMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure based on your needs
)


# Utility functions
def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{int(time.time() * 1000)}_{hash(time.time()) % 10000}"


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def log_request(request_id: str, endpoint: str, request_data: Dict[str, Any]):
    """Log request information."""
    logger.info(f"Request {request_id} to {endpoint}: {json.dumps(request_data, default=str)}")


def log_response(request_id: str, response_data: Dict[str, Any], duration: float):
    """Log response information."""
    logger.info(f"Response {request_id} completed in {duration:.2f}s: {json.dumps(response_data, default=str)}")


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token if authentication is enabled."""
    if not credentials:
        return None
    
    # Add your token verification logic here
    # For now, just return the token
    return credentials.credentials


# API endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic information."""
    return """
    <html>
        <head>
            <title>Embedding-to-Prompt API</title>
        </head>
        <body>
            <h1>Embedding-to-Prompt Generation API</h1>
            <p>Welcome to the Embedding-to-Prompt Generation API!</p>
            <ul>
                <li><a href="/docs">API Documentation (Swagger)</a></li>
                <li><a href="/redoc">API Documentation (ReDoc)</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
        </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global request_count, start_time, inference_engine
    
    return HealthResponse(
        status="healthy" if inference_engine else "unhealthy",
        timestamp=get_current_timestamp(),
        uptime=time.time() - start_time,
        model_loaded=inference_engine is not None,
        requests_processed=request_count,
        version="1.0.0"
    )


@app.post("/generate_prompt", response_model=PromptResponse)
@limiter.limit("60/minute")
async def generate_prompt(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    token: str = Depends(verify_token)
):
    """
    Generate a prompt from a target embedding.
    
    This endpoint takes a target embedding vector and generates an optimized prompt
    that, when used with an LLM, should produce output with an embedding similar 
    to the target embedding.
    """
    global request_count, inference_engine
    
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        # Validate inference engine
        if not inference_engine:
            raise HTTPException(
                status_code=503,
                detail="Inference engine not initialized"
            )
        
        # Log request
        log_request(request_id, "/generate_prompt", {
            "embedding_dim": len(request.target_embedding),
            "generation_params": request.generation_params,
            "return_metadata": request.return_metadata
        })
        
        # Generate prompt
        prompt = inference_engine.generate_prompt(
            request.target_embedding,
            request.generation_params or {}
        )
        
        # Prepare metadata
        metadata = None
        if request.return_metadata:
            metadata = {
                "model_architecture": config['model']['architecture'],
                "generation_time": time.time() - start_time,
                "prompt_length": len(prompt.split()),
                "embedding_dim": len(request.target_embedding)
            }
        
        # Update request count
        request_count += 1
        
        # Log response
        duration = time.time() - start_time
        background_tasks.add_task(log_response, request_id, {"prompt_length": len(prompt.split())}, duration)
        
        return PromptResponse(
            prompt=prompt,
            metadata=metadata,
            timestamp=get_current_timestamp(),
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Error in generate_prompt {request_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/generate_prompt_from_text", response_model=PromptResponse)
@limiter.limit("30/minute")
async def generate_prompt_from_text(
    request: TextRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    token: str = Depends(verify_token)
):
    """
    Generate a prompt from input text.
    
    This endpoint takes input text, embeds it using OpenAI's embedding model,
    and then generates an optimized prompt from that embedding.
    """
    global request_count, inference_engine
    
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        # Validate inference engine
        if not inference_engine:
            raise HTTPException(
                status_code=503,
                detail="Inference engine not initialized"
            )
        
        # Log request
        log_request(request_id, "/generate_prompt_from_text", {
            "text_length": len(request.text),
            "generation_params": request.generation_params,
            "return_metadata": request.return_metadata
        })
        
        # Generate prompt
        prompt = inference_engine.generate_prompt_from_text(
            request.text,
            request.generation_params or {}
        )
        
        # Prepare metadata
        metadata = None
        if request.return_metadata:
            metadata = {
                "model_architecture": config['model']['architecture'],
                "generation_time": time.time() - start_time,
                "prompt_length": len(prompt.split()),
                "input_text_length": len(request.text)
            }
        
        # Update request count
        request_count += 1
        
        # Log response
        duration = time.time() - start_time
        background_tasks.add_task(log_response, request_id, {"prompt_length": len(prompt.split())}, duration)
        
        return PromptResponse(
            prompt=prompt,
            metadata=metadata,
            timestamp=get_current_timestamp(),
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Error in generate_prompt_from_text {request_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/generate_prompts_batch", response_model=BatchResponse)
@limiter.limit("10/minute")
async def generate_prompts_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    token: str = Depends(verify_token)
):
    """
    Generate prompts from multiple embeddings in batch.
    
    This endpoint processes multiple embeddings at once for efficient batch processing.
    """
    global request_count, inference_engine
    
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        # Validate inference engine
        if not inference_engine:
            raise HTTPException(
                status_code=503,
                detail="Inference engine not initialized"
            )
        
        # Log request
        log_request(request_id, "/generate_prompts_batch", {
            "batch_size": len(request.embeddings),
            "generation_params": request.generation_params,
            "return_metadata": request.return_metadata
        })
        
        # Generate prompts
        prompts = inference_engine.generate_prompts_batch(
            request.embeddings,
            request.generation_params or {}
        )
        
        # Prepare metadata
        metadata = None
        if request.return_metadata:
            metadata = {
                "model_architecture": config['model']['architecture'],
                "generation_time": time.time() - start_time,
                "batch_size": len(request.embeddings),
                "avg_prompt_length": np.mean([len(p.split()) for p in prompts])
            }
        
        # Update request count
        request_count += len(request.embeddings)
        
        # Log response
        duration = time.time() - start_time
        background_tasks.add_task(log_response, request_id, {"batch_size": len(prompts)}, duration)
        
        return BatchResponse(
            prompts=prompts,
            metadata=metadata,
            timestamp=get_current_timestamp(),
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Error in generate_prompts_batch {request_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/generate_prompts_parallel", response_model=ParallelBatchResponse)
@limiter.limit("5/minute")
async def generate_prompts_parallel(
    request: ParallelBatchRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    token: str = Depends(verify_token)
):
    """
    Generate prompts from multiple embeddings with parallel processing and evaluation.
    
    This endpoint uses advanced parallel processing with OpenAI rate limiting,
    automatic quality evaluation, and optional prompt refinement.
    """
    global request_count, inference_engine
    
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        # Validate inference engine
        if not inference_engine:
            raise HTTPException(
                status_code=503,
                detail="Inference engine not initialized"
            )
        
        # Log request
        log_request(request_id, "/generate_prompts_parallel", {
            "batch_size": len(request.embeddings),
            "max_concurrent": request.max_concurrent,
            "rate_limit_tier": request.rate_limit_tier,
            "evaluate_quality": request.evaluate_quality,
            "generate_variants": request.generate_variants
        })
        
        # Map rate limit tier string to enum
        tier_map = {
            "tier_1": RateLimitTier.TIER_1,
            "tier_2": RateLimitTier.TIER_2,
            "tier_3": RateLimitTier.TIER_3,
            "tier_4": RateLimitTier.TIER_4,
            "tier_5": RateLimitTier.TIER_5,
        }
        
        rate_limit_tier = tier_map.get(request.rate_limit_tier, RateLimitTier.TIER_2)
        
        # Initialize parallel processor
        parallel_generator = ParallelPromptGenerator(
            inference_engine=inference_engine,
            config=config,
            rate_limit_tier=rate_limit_tier
        )
        
        try:
            # Process embeddings in parallel
            if request.evaluate_quality:
                results = await parallel_generator.generate_prompts_with_evaluation(
                    request.embeddings,
                    max_concurrent=request.max_concurrent
                )
            else:
                # Simple batch processing without evaluation
                np_embeddings = [np.array(emb) for emb in request.embeddings]
                initial_prompts = []
                for embedding in np_embeddings:
                    prompt = inference_engine.generate_prompt(embedding)
                    initial_prompts.append(prompt)
                
                results = [
                    {
                        "embedding_index": i,
                        "original_prompt": prompt,
                        "original_result": {"success": True, "similarity": None},
                        "variants": [],
                        "best_variant": None
                    }
                    for i, prompt in enumerate(initial_prompts)
                ]
            
            # Calculate summary statistics
            processing_time = time.time() - start_time
            
            successful_results = [r for r in results if r.get('original_result', {}).get('success', False)]
            similarities = [
                r.get('original_result', {}).get('similarity', 0) 
                for r in successful_results 
                if r.get('original_result', {}).get('similarity') is not None
            ]
            
            summary = {
                "total_processed": len(results),
                "successful": len(successful_results),
                "failed": len(results) - len(successful_results),
                "processing_time": processing_time,
                "average_similarity": np.mean(similarities) if similarities else None,
                "max_similarity": max(similarities) if similarities else None,
                "min_similarity": min(similarities) if similarities else None,
                "variants_generated": sum(1 for r in results if r.get('variants')),
                "concurrent_workers": request.max_concurrent,
                "rate_limit_tier": request.rate_limit_tier
            }
            
            # Prepare metadata
            metadata = None
            if request.return_metadata:
                metadata = {
                    "model_architecture": config['model']['architecture'],
                    "processing_details": {
                        "parallel_processing": True,
                        "rate_limiting": True,
                        "quality_evaluation": request.evaluate_quality,
                        "variant_generation": request.generate_variants
                    },
                    "performance_metrics": {
                        "requests_per_second": len(results) / processing_time if processing_time > 0 else 0,
                        "average_processing_time": processing_time / len(results) if results else 0
                    }
                }
            
            # Update request count
            request_count += len(request.embeddings)
            
            # Log response
            background_tasks.add_task(
                log_response, 
                request_id, 
                {
                    "batch_size": len(results),
                    "successful": len(successful_results),
                    "processing_time": processing_time
                }, 
                processing_time
            )
            
            return ParallelBatchResponse(
                results=results,
                summary=summary,
                metadata=metadata,
                timestamp=get_current_timestamp(),
                request_id=request_id
            )
            
        finally:
            # Clean up resources
            await parallel_generator.close()
        
    except Exception as e:
        logger.error(f"Error in generate_prompts_parallel {request_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """Get API metrics."""
    global request_count, start_time
    
    return {
        "requests_processed": request_count,
        "uptime": time.time() - start_time,
        "requests_per_second": request_count / (time.time() - start_time) if time.time() > start_time else 0,
        "model_loaded": inference_engine is not None,
        "timestamp": get_current_timestamp()
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            request_id=generate_request_id(),
            timestamp=get_current_timestamp()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            request_id=generate_request_id(),
            timestamp=get_current_timestamp()
        ).dict()
    )


# Run the application
if __name__ == "__main__":
    # Configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    workers = int(os.getenv('WORKERS', 1))
    
    # Run with uvicorn
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        reload=os.getenv('RELOAD', 'false').lower() == 'true'
    ) 