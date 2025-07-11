# Parallel Embedding-to-Prompt Processing

This system provides advanced parallel processing capabilities for embedding-to-prompt generation with built-in OpenAI rate limiting, quality evaluation, and performance optimization.

## Features

- **Parallel Processing**: Process multiple embeddings simultaneously with configurable concurrency
- **Rate Limiting**: Automatic OpenAI API rate limiting based on your tier
- **Quality Evaluation**: Automatic prompt quality assessment using embedding similarity
- **Retry Logic**: Robust error handling with exponential backoff
- **Performance Metrics**: Detailed performance and quality analytics
- **Scalable Architecture**: Designed to handle large batches efficiently

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"

# Optional
export API_BASE_URL="http://localhost:8000"
export API_KEY="your-api-key"  # If using API authentication
```

### 3. Start the API Server

```bash
python api/app.py
```

### 4. Run the Example

```bash
python run_parallel_example.py
```

## API Endpoints

### `/generate_prompts_parallel`

Advanced parallel processing endpoint with quality evaluation.

**Request:**
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "max_concurrent": 10,
  "rate_limit_tier": "tier_2",
  "evaluate_quality": true,
  "generate_variants": false,
  "return_metadata": true
}
```

**Response:**
```json
{
  "results": [
    {
      "embedding_index": 0,
      "original_prompt": "Generated prompt text...",
      "original_result": {
        "similarity": 0.85,
        "success": true,
        "processing_time": 2.3
      },
      "variants": [],
      "best_variant": null
    }
  ],
  "summary": {
    "total_processed": 10,
    "successful": 10,
    "failed": 0,
    "processing_time": 15.2,
    "average_similarity": 0.78,
    "max_similarity": 0.92,
    "min_similarity": 0.64
  },
  "metadata": {
    "performance_metrics": {
      "requests_per_second": 0.66,
      "average_processing_time": 1.52
    }
  }
}
```

## Rate Limiting

The system supports OpenAI's rate limiting tiers:

| Tier | Requests/Min | Requests/Day | Concurrent |
|------|-------------|-------------|-----------|
| tier_1 | 3 | 200 | 3 |
| tier_2 | 3,500 | 10,000 | 50 |
| tier_3 | 5,000 | 10,000 | 100 |
| tier_4 | 10,000 | 30,000 | 200 |
| tier_5 | 30,000 | 100,000 | 500 |

## Configuration Options

### Parallel Processing Parameters

- `max_concurrent`: Maximum number of concurrent requests (1-50)
- `rate_limit_tier`: OpenAI rate limit tier (tier_1 through tier_5)
- `evaluate_quality`: Whether to evaluate prompt quality using OpenAI
- `generate_variants`: Whether to generate refined prompt variants
- `return_metadata`: Whether to return detailed processing metadata

### Quality Evaluation

When `evaluate_quality` is enabled, the system:
1. Generates text using the created prompt
2. Creates embeddings for the generated text
3. Computes cosine similarity with the target embedding
4. Provides quality scores and performance metrics

## Performance Comparison

The parallel processing system provides significant performance improvements:

```
Processing 10 embeddings:
- Sequential: ~45 seconds
- Parallel (5 workers): ~12 seconds
- Speedup: ~3.8x faster
```

## Advanced Usage

### Using the Python API Directly

```python
from scripts.async_processing import ParallelPromptGenerator, RateLimitTier
from scripts.inference import EmbeddingToPromptInference
import asyncio
import numpy as np

# Initialize components
inference_engine = EmbeddingToPromptInference("models/best_model.pt")
generator = ParallelPromptGenerator(
    inference_engine=inference_engine,
    config=config,
    rate_limit_tier=RateLimitTier.TIER_2
)

# Generate embeddings
embeddings = [np.random.random(1024) for _ in range(10)]

# Process in parallel
async def process_batch():
    results = await generator.generate_prompts_with_evaluation(
        embeddings,
        max_concurrent=5
    )
    return results

# Run the async function
results = asyncio.run(process_batch())
```

### Custom Rate Limiting

```python
from scripts.async_processing import RateLimitConfig, OpenAIRateLimiter

# Create custom rate limiting configuration
config = RateLimitConfig(
    requests_per_minute=100,
    requests_per_day=1000,
    tokens_per_minute=10000,
    tokens_per_day=100000,
    concurrent_requests=10
)

rate_limiter = OpenAIRateLimiter(config)
```

## Error Handling

The system includes robust error handling:

- **Automatic Retries**: Failed requests are retried with exponential backoff
- **Rate Limit Compliance**: Automatic waiting when rate limits are reached
- **Graceful Degradation**: Individual failures don't stop the entire batch
- **Detailed Logging**: Comprehensive error reporting and debugging information

## Monitoring and Metrics

### Key Metrics Tracked

- **Processing Time**: Total and per-request processing time
- **Success Rate**: Percentage of successful requests
- **Quality Scores**: Embedding similarity metrics
- **Rate Limit Usage**: Current rate limit status
- **Throughput**: Requests per second

### Performance Optimization Tips

1. **Choose Appropriate Concurrency**: Start with 5-10 concurrent requests
2. **Monitor Rate Limits**: Use the appropriate tier for your usage
3. **Batch Size**: Process 10-50 embeddings per batch for optimal performance
4. **Quality vs Speed**: Disable quality evaluation for faster processing

## Testing

### Run the Comprehensive Test Suite

```bash
python test_parallel_processing.py
```

This will:
- Generate 10 test embeddings
- Process them in parallel
- Compare with sequential processing
- Display performance metrics
- Show sample results

### Run the Quick Example

```bash
python run_parallel_example.py
```

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**: Ensure your OpenAI API key has sufficient quota
2. **Timeout Errors**: Increase timeout values for large batches
3. **Memory Issues**: Reduce batch size or concurrency for large embeddings
4. **API Connection**: Verify the API server is running and accessible

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Architecture

The system consists of several key components:

1. **AsyncEmbeddingProcessor**: Core async processing engine
2. **OpenAIRateLimiter**: Intelligent rate limiting
3. **ParallelPromptGenerator**: High-level interface
4. **FastAPI Endpoints**: REST API interface

## Contributing

When adding new features:

1. Maintain async/await patterns
2. Include proper error handling
3. Add comprehensive logging
4. Update rate limiting logic as needed
5. Include performance metrics

## License

This project is licensed under the MIT License. 