#!/usr/bin/env python3
"""
Simple test script to generate embeddings for first 10 outputs and upload to Pinecone
"""

import csv
import os
from dotenv import load_dotenv
from openai import OpenAI
import time

# Load environment variables
load_dotenv()

def main():
    print("🧪 Testing Embedding Generation for First 10 Outputs")
    print("=" * 60)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
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
    
    # Generate embeddings
    print("\n🔄 Generating embeddings...")
    embeddings = []
    
    for i, output in enumerate(outputs):
        print(f"  Processing {i+1}/10: {output[:50]}...")
        
        try:
            response = client.embeddings.create(
                input=output,
                model='text-embedding-3-small'
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
            
            # Small delay to be nice to API
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Error generating embedding {i+1}: {e}")
            return
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    
    # Try to upload to Pinecone (optional)
    pinecone_key = os.getenv('PINECONE_API_KEY')
    if pinecone_key and pinecone_key != 'your_pinecone_api_key_here':
        print("\n☁️ Uploading to Pinecone...")
        try:
            from pinecone import Pinecone
            
            pc = Pinecone(api_key=pinecone_key)
            
            # Create or connect to index
            index_name = "vector-to-prompt"
            if index_name not in pc.list_indexes().names():
                print(f"Creating Pinecone index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=1536,  # text-embedding-3-small dimensions
                    metric="cosine"
                )
                time.sleep(5)  # Wait for index to be ready
            
            index = pc.Index(index_name)
            
            # Prepare vectors for upload
            vectors = []
            for i, (embedding, prompt, output) in enumerate(zip(embeddings, prompts, outputs)):
                vectors.append({
                    "id": f"test-{i}",
                    "values": embedding,
                    "metadata": {
                        "prompt": prompt[:500],  # Truncate for metadata limits
                        "output": output[:500],  # Truncate for metadata limits
                        "type": "test"
                    }
                })
            
            # Upload to Pinecone
            index.upsert(vectors=vectors)
            print(f"✅ Uploaded {len(vectors)} vectors to Pinecone!")
            
            # Test query
            query_result = index.query(
                vector=embeddings[0],
                top_k=3,
                include_metadata=True
            )
            print(f"🔍 Test query returned {len(query_result.matches)} matches")
            
        except ImportError:
            print("⚠️ Pinecone not available, skipping upload")
        except Exception as e:
            print(f"⚠️ Pinecone upload failed: {e}")
    else:
        print("\n⚠️ Pinecone API key not set, skipping upload")
    
    print("\n🎉 Test completed successfully!")
    print(f"📊 Summary:")
    print(f"   - Processed: 10 outputs")
    print(f"   - Generated: {len(embeddings)} embeddings")
    print(f"   - Dimensions: {len(embeddings[0]) if embeddings else 'N/A'}")
    print(f"   - Cost estimate: ${(10 * 8 / 1_000_000) * 0.02:.6f}")

if __name__ == "__main__":
    main() 