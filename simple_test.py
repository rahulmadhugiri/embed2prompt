#!/usr/bin/env python3
import sys
import os
import json
import time
import requests
from typing import Dict, Any

sys.path.append('.')

API_BASE_URL = "http://localhost:8000"


def test_api_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def test_single_prompt_generation() -> Dict[str, Any]:
    test_embedding = [0.1] * 1024  # Simple test embedding
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate_prompt",
            json={
                "target_embedding": test_embedding,
                "return_metadata": True
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def test_text_to_prompt() -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate_prompt_from_text",
            json={
                "text": "How to build a web application",
                "return_metadata": True
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def test_batch_processing() -> Dict[str, Any]:
    test_embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate_prompts_batch",
            json={
                "embeddings": test_embeddings,
                "return_metadata": True
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    print("Testing Embedding-to-Prompt API")
    
    # Test API health
    if not test_api_health():
        print("❌ API health check failed. Please start the server.")
        return
    print("✅ API health check passed")
    
    # Test single prompt generation
    result = test_single_prompt_generation()
    if result["success"]:
        prompt = result["data"]["prompt"]
        print(f"✅ Single prompt: {prompt[:60]}...")
    else:
        print(f"❌ Single prompt failed: {result['error']}")
    
    # Test text to prompt
    result = test_text_to_prompt()
    if result["success"]:
        prompt = result["data"]["prompt"]
        print(f"✅ Text to prompt: {prompt[:60]}...")
    else:
        print(f"❌ Text to prompt failed: {result['error']}")
    
    # Test batch processing
    result = test_batch_processing()
    if result["success"]:
        successful = result["data"]["successful"]
        total = result["data"]["total_processed"]
        print(f"✅ Batch processing: {successful}/{total} successful")
    else:
        print(f"❌ Batch processing failed: {result['error']}")
    
    print("Testing completed!")


if __name__ == "__main__":
    main() 