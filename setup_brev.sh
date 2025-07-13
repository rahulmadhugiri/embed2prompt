#!/bin/bash
# NVIDIA Brev Setup Script for Embeddings-to-Prompts Training

echo "🚀 Setting up NVIDIA Brev environment for embeddings-to-prompts training..."

# Update system
apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux

# Install Python 3.9+
apt-get install -y python3.9 python3.9-pip python3.9-venv
ln -sf /usr/bin/python3.9 /usr/bin/python
ln -sf /usr/bin/pip3.9 /usr/bin/pip

# Create virtual environment
python -m venv /opt/venv
source /opt/venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt

# Additional ML/AI packages
pip install \
    accelerate \
    transformers[torch] \
    datasets \
    evaluate \
    tensorboard \
    wandb \
    jupyter \
    ipywidgets

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace:$PYTHONPATH

# Create directories
mkdir -p /workspace/logs
mkdir -p /workspace/models/saved
mkdir -p /workspace/data/embeddings

# Set up Jupyter
jupyter nbextension enable --py widgetsnbextension --sys-prefix

echo "✅ Environment setup complete!"
echo "📊 Training data: $(wc -l < /workspace/data/embeddings/training_data.jsonl) examples"
echo "🎯 Ready to train with: python scripts/train.py" 