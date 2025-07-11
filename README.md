# Embedding-to-Prompt Generation API

This project generates optimized prompts from target embeddings using a trained neural network model. Given an embedding vector, it creates prompts that should produce similar embeddings when used with language models.

## The Future of Personalized AI

The next frontier in consumer technology lies at the intersection of semantic understanding and generative intelligence. Today's digital landscape is defined by a fundamental architectural tension: recommendation systems operate in high-dimensional embedding spaces that capture the nuanced topology of human preferences, while the most powerful content generation models (large language models) interface exclusively through natural language prompts. This creates a critical gap in the infrastructure of personalized experiences.

Consider the trajectory of modern consumer products: every platform, from social media to e-commerce, relies on embeddings to model user intent, content similarity, and behavioral patterns. When these systems need to generate new content—whether for dynamic feeds, personalized recommendations, or adaptive interfaces—they must abandon this rich geometric understanding and resort to crude prompt engineering or generic content retrieval.

**The Missing Link**

This project introduces the concept of a "latent prompt space": a learned translation layer that bridges vector representations of intent with the natural language interfaces that drive today's most capable AI systems. By training neural networks to map arbitrary points in embedding space to optimized prompts, this approach makes it possible to generate content that lands precisely where intended in the semantic landscape of user preferences.

The implications extend far beyond technical elegance. This architecture enables a new class of AI systems that are simultaneously:

- **Geometrically aware**: Operating with full knowledge of the embedding space topology
- **Generatively capable**: Leveraging the full power of large language models  
- **Continuously adaptive**: Evolving through feedback loops that refine both understanding and generation
- **Architecturally modular**: Separating content creation from preference modeling

**Toward Semantic-Native Experiences**

The convergence of these capabilities points toward a fundamental shift in how digital products understand and serve human needs. Rather than choosing between broad personalization and precise generation, future systems will navigate smoothly across the entire spectrum of user intent, from broad preferences captured in embedding space to specific, contextual content generation driven by optimized prompts.

This represents more than an incremental improvement in recommendation systems or content generation. It's an architectural foundation for AI that truly understands the geometric structure of human preferences and can generate accordingly, making possible experiences that are simultaneously deeply personal and infinitely creative.

## Setup

### Requirements

- Python 3.8+
- OpenAI API key
- At least 2GB RAM for model loading

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd embeddings-to-prompts
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

4. Start the API server:
```bash
python api/app.py
```

The API will be available at `http://localhost:8000`

## Usage

### Single Prompt Generation

```bash
curl -X POST "http://localhost:8000/generate_prompt" \
  -H "Content-Type: application/json" \
  -d '{
    "target_embedding": [0.1, 0.2, ...],
    "return_metadata": true
  }'
```

### Parallel Batch Processing

For processing multiple embeddings efficiently:

```bash
curl -X POST "http://localhost:8000/generate_prompts_parallel" \
  -H "Content-Type: application/json" \
  -d '{
    "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    "max_concurrent": 5,
    "evaluate_quality": true
  }'
```

### Quick Test

Run the example script to see it in action:

```bash
python run_parallel_example.py
```

## API Endpoints

- `GET /health` - Check if the service is running
- `POST /generate_prompt` - Generate a single prompt
- `POST /generate_prompt_from_text` - Generate prompt from input text
- `POST /generate_prompts_batch` - Process multiple embeddings
- `POST /generate_prompts_parallel` - Parallel processing with quality evaluation

See `/docs` for full API documentation.

## Configuration

Key settings in `config.yaml`:

- `model.architecture`: Model type (currently supports t5-small)
- `openai.embedding_model`: OpenAI embedding model to use
- `api.rate_limit`: Requests per minute limit

## File Structure

```
├── api/app.py                 # FastAPI server
├── scripts/
│   ├── async_processing.py    # Parallel processing engine
│   ├── inference.py           # Model inference
│   └── train.py              # Model training
├── models/architecture.py     # Neural network models
├── config.yaml               # Configuration
├── requirements.txt          # Dependencies
└── tests/                    # Test scripts
```

## Training Your Own Model

If you have a dataset of prompt-output pairs:

1. Prepare your data in CSV format with 'prompt' and 'output' columns
2. Generate embeddings: `python scripts/embed_outputs.py`
3. Train the model: `python scripts/train.py`

## Contributing

This is a research project. Feel free to experiment with different model architectures or training approaches.

## License

MIT License 