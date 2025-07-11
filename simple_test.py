#!/usr/bin/env python3
"""
Simple test using requests to call OpenAI API directly
"""

import csv
import json
import os
import requests
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def main():
    print("🧪 Testing Embedding Generation with Direct API Calls")
    print("=" * 60)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your_openai_api_key_here':
        print("❌ OpenAI API key not found in .env file")
        return
    
    # Read first 10 outputs from dataset
    outputs = []
    prompts = []
    
    print("📖 Reading dataset...")
    with open('data/alpaca_data.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i >= 10:  # Only first 10
                break
            outputs.append(row['output'])
            prompts.append(row['prompt'])
    
    print(f"✅ Loaded {len(outputs)} outputs")
    
    # Generate embeddings using direct API calls
    print("\n🔄 Generating embeddings...")
    embeddings = []
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    for i, output in enumerate(outputs):
        print(f"  Processing {i+1}/10: {output[:50]}...")
        
        try:
            response = requests.post(
                'https://api.openai.com/v1/embeddings',
                headers=headers,
                json={
                    'input': output,
                    'model': 'text-embedding-3-small',
                    'dimensions': 1024  # Reduce to 1024 to match Pinecone index
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data['data'][0]['embedding']
                embeddings.append({
                    'id': f'test-{i}',
                    'embedding': embedding,
                    'prompt': prompts[i],
                    'output': output,
                    'tokens': data.get('usage', {}).get('total_tokens', 0)
                })
                print(f"    ✅ Generated {len(embedding)}-dimensional embedding")
            else:
                print(f"    ❌ API error: {response.status_code} - {response.text}")
                return
            
            # Small delay to be nice to API
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Error generating embedding {i+1}: {e}")
            return
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    
    # Save embeddings to file
    print("\n💾 Saving embeddings to file...")
    with open('test_embeddings.json', 'w') as f:
        json.dump(embeddings, f, indent=2)
    print("✅ Embeddings saved to test_embeddings.json")
    
    # Try to upload to Pinecone (optional)
    pinecone_key = os.getenv('PINECONE_API_KEY')
    if pinecone_key and pinecone_key != 'your_pinecone_api_key_here':
        print("\n☁️ Uploading to Pinecone...")
        try:
            from pinecone import Pinecone
            
            pc = Pinecone(api_key=pinecone_key)
            
            # Use existing index
            index_name = "vector-to-prompt"
            index = pc.Index(index_name)
            
            # Prepare vectors for upload
            vectors = []
            for item in embeddings:
                vectors.append({
                    "id": item['id'],
                    "values": item['embedding'],
                    "metadata": {
                        "prompt": item['prompt'][:500],  # Truncate for metadata limits
                        "output": item['output'][:500],  # Truncate for metadata limits
                        "type": "test",
                        "tokens": item['tokens']
                    }
                })
            
            # Upload to Pinecone
            index.upsert(vectors=vectors)
            print(f"✅ Uploaded {len(vectors)} vectors to Pinecone!")
            
            # Test query
            query_result = index.query(
                vector=embeddings[0]['embedding'],
                top_k=3,
                include_metadata=True
            )
            print(f"🔍 Test query returned {len(query_result.matches)} matches")
            for match in query_result.matches:
                print(f"   Score: {match.score:.4f} - {match.metadata.get('prompt', 'No prompt')[:100]}...")
            
        except ImportError:
            print("⚠️ Pinecone not available, skipping upload")
        except Exception as e:
            print(f"⚠️ Pinecone upload failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️ Pinecone API key not set, skipping upload")
    
    # Calculate cost
    total_tokens = sum(item['tokens'] for item in embeddings)
    cost = (total_tokens / 1_000_000) * 0.02
    
    print("\n🎉 Test completed successfully!")
    print(f"📊 Summary:")
    print(f"   - Processed: {len(embeddings)} outputs")
    print(f"   - Generated: {len(embeddings)} embeddings")
    print(f"   - Dimensions: {len(embeddings[0]['embedding']) if embeddings else 'N/A'}")
    print(f"   - Total tokens: {total_tokens}")
    print(f"   - Actual cost: ${cost:.6f}")

if __name__ == "__main__":
    main() 