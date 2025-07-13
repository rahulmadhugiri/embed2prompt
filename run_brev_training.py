#!/usr/bin/env python3
"""
NVIDIA Brev Training Launcher
Handles the complete training pipeline on GPU instances.
"""

import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brev_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_gpu():
    """Check if GPU is available and working."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ GPU Available: {gpu_name}")
            logger.info(f"💾 GPU Memory: {gpu_memory:.1f}GB")
            return True
        else:
            logger.error("❌ No GPU available!")
            return False
    except ImportError:
        logger.error("❌ PyTorch not installed!")
        return False

def verify_training_data():
    """Verify training data exists and is valid."""
    data_path = Path("data/embeddings/training_data.jsonl")
    
    if not data_path.exists():
        logger.error(f"❌ Training data not found: {data_path}")
        return False
    
    # Count lines and verify format
    try:
        with open(data_path, 'r') as f:
            lines = f.readlines()
            total_examples = len(lines)
            
            # Verify first line is valid JSON
            first_example = json.loads(lines[0])
            required_fields = ['embedding', 'prompt', 'output']
            
            for field in required_fields:
                if field not in first_example:
                    logger.error(f"❌ Missing field '{field}' in training data")
                    return False
            
            embedding_dim = len(first_example['embedding'])
            logger.info(f"✅ Training data: {total_examples:,} examples")
            logger.info(f"📊 Embedding dimension: {embedding_dim}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error reading training data: {e}")
        return False

def setup_environment():
    """Set up the training environment."""
    logger.info("🔧 Setting up training environment...")
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Create necessary directories
    dirs_to_create = [
        'logs',
        'models/saved',
        'data/embeddings'
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {dir_path}")

def run_training():
    """Execute the training process."""
    logger.info("🚀 Starting training process...")
    
    # Training command
    cmd = [
        sys.executable,
        'scripts/train.py',
        '--gpu',
        '--save-model',
        '--tensorboard'
    ]
    
    logger.info(f"🔨 Running command: {' '.join(cmd)}")
    
    # Start training
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream output
    for line in iter(process.stdout.readline, ''):
        logger.info(line.rstrip())
    
    process.wait()
    end_time = time.time()
    
    if process.returncode == 0:
        training_time = end_time - start_time
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"⏱️  Total training time: {training_time/60:.1f} minutes")
        return True
    else:
        logger.error(f"❌ Training failed with return code: {process.returncode}")
        return False

def save_training_summary():
    """Save a summary of the training session."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'gpu_info': get_gpu_info(),
        'training_data': get_training_data_info(),
        'model_config': get_model_config()
    }
    
    summary_path = Path('logs/training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📝 Training summary saved to: {summary_path}")

def get_gpu_info():
    """Get GPU information."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'name': torch.cuda.get_device_name(0),
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'cuda_version': torch.version.cuda
            }
    except:
        pass
    return None

def get_training_data_info():
    """Get training data information."""
    data_path = Path("data/embeddings/training_data.jsonl")
    if data_path.exists():
        with open(data_path, 'r') as f:
            lines = f.readlines()
            return {
                'total_examples': len(lines),
                'file_size_mb': data_path.stat().st_size / (1024*1024)
            }
    return None

def get_model_config():
    """Get model configuration."""
    config_path = Path('config.yaml')
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None

def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("🎯 NVIDIA Brev Training Pipeline Started")
    logger.info("=" * 60)
    
    # Step 1: Check GPU
    if not check_gpu():
        logger.error("❌ GPU check failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Verify training data
    if not verify_training_data():
        logger.error("❌ Training data verification failed. Exiting.")
        sys.exit(1)
    
    # Step 3: Setup environment
    setup_environment()
    
    # Step 4: Run training
    if run_training():
        logger.info("🎉 Training pipeline completed successfully!")
        save_training_summary()
        
        # Show model location
        model_path = Path('models/saved/best_model.pt')
        if model_path.exists():
            logger.info(f"💾 Trained model saved to: {model_path}")
        
        logger.info("🚀 Ready for inference!")
        
    else:
        logger.error("❌ Training pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 