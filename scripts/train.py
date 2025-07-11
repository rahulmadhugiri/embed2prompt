#!/usr/bin/env python3
"""
Training Script for Embedding-to-Prompt Model
Implements supervised learning pipeline for (target_embedding → prompt) pairs.
"""

import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import random
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from transformers import T5Tokenizer, get_linear_schedule_with_warmup
import jsonlines
import pandas as pd

# Import our model architecture
import sys
sys.path.append('.')
from models.architecture import EmbeddingToPromptModel
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingPromptDataset(Dataset):
    """Dataset for embedding-to-prompt pairs."""
    
    def __init__(self, data_path: str, tokenizer: T5Tokenizer, 
                 max_prompt_length: int = 512, split: str = 'train'):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the training data file
            tokenizer: Tokenizer for encoding prompts
            max_prompt_length: Maximum prompt length
            split: Data split (train/val/test)
        """
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.split = split
        
        # Load data
        self.data = self._load_data(data_path)
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load training data from file."""
        data = []
        
        if data_path.endswith('.jsonl'):
            with jsonlines.open(data_path, 'r') as reader:
                for item in reader:
                    data.append(item)
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        item = self.data[idx]
        
        # Get embedding
        embedding = torch.tensor(item['embedding'], dtype=torch.float32)
        
        # Get prompt
        prompt = item['prompt']
        
        # Tokenize prompt
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_prompt_length,
            return_tensors='pt'
        )
        
        # Prepare input and target tokens
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For T5, we need decoder input ids (shifted right)
        decoder_input_ids = torch.zeros_like(input_ids)
        decoder_input_ids[1:] = input_ids[:-1]
        decoder_input_ids[0] = self.tokenizer.pad_token_id
        
        return {
            'embedding': embedding,
            'input_ids': input_ids,
            'decoder_input_ids': decoder_input_ids,
            'attention_mask': attention_mask,
            'prompt': prompt
        }


def create_data_splits(data_path: str, config: Dict[str, Any]) -> Tuple[str, str, str]:
    """Create train/val/test splits from the data."""
    
    # Load data
    if data_path.endswith('.jsonl'):
        data = []
        with jsonlines.open(data_path, 'r') as reader:
            for item in reader:
                data.append(item)
    else:
        with open(data_path, 'r') as f:
            data = json.load(f)
    
    # Shuffle data
    random.shuffle(data)
    
    # Calculate split sizes
    total_samples = len(data)
    train_size = int(total_samples * config['data']['train_split'])
    val_size = int(total_samples * config['data']['val_split'])
    
    # Split data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Create output directory
    output_dir = Path(data_path).parent / 'splits'
    output_dir.mkdir(exist_ok=True)
    
    # Save splits
    train_path = output_dir / 'train.jsonl'
    val_path = output_dir / 'val.jsonl'
    test_path = output_dir / 'test.jsonl'
    
    for data_split, path in [(train_data, train_path), (val_data, val_path), (test_data, test_path)]:
        with jsonlines.open(path, 'w') as writer:
            for item in data_split:
                writer.write(item)
    
    logger.info(f"Created data splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    return str(train_path), str(val_path), str(test_path)


def compute_metrics(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    from collections import Counter
    import nltk
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
    except ImportError:
        logger.warning("NLTK not available, skipping BLEU score computation")
        return {}
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    metrics = {}
    
    # BLEU score
    bleu_scores = []
    for pred, target in zip(predictions, targets):
        try:
            pred_tokens = word_tokenize(pred.lower())
            target_tokens = word_tokenize(target.lower())
            
            if len(pred_tokens) > 0 and len(target_tokens) > 0:
                bleu = sentence_bleu([target_tokens], pred_tokens)
                bleu_scores.append(bleu)
        except Exception:
            continue
    
    if bleu_scores:
        metrics['bleu'] = np.mean(bleu_scores)
    
    # Exact match
    exact_matches = sum(1 for pred, target in zip(predictions, targets) if pred.strip() == target.strip())
    metrics['exact_match'] = exact_matches / len(predictions)
    
    # Length statistics
    pred_lengths = [len(pred.split()) for pred in predictions]
    target_lengths = [len(target.split()) for target in targets]
    
    metrics['avg_pred_length'] = np.mean(pred_lengths)
    metrics['avg_target_length'] = np.mean(target_lengths)
    metrics['length_ratio'] = np.mean(pred_lengths) / np.mean(target_lengths)
    
    return metrics


def evaluate_model(model: EmbeddingToPromptModel, dataloader: DataLoader, 
                  device: torch.device, config: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate the model on a dataset."""
    model.eval()
    
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            embeddings = batch['embedding'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            if config['model']['architecture'] == 't5-small':
                decoder_input_ids = batch['decoder_input_ids'].to(device)
                
                # Forward pass
                outputs = model(embeddings, decoder_input_ids, attention_mask)
                
                # Compute loss
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    input_ids.view(-1),
                    ignore_index=model.tokenizer.pad_token_id
                )
                
            else:
                # For MLP and Transformer models
                outputs = model(embeddings, input_ids)
                
                # Compute loss
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    input_ids.view(-1),
                    ignore_index=model.tokenizer.pad_token_id
                )
            
            total_loss += loss.item()
            
            # Generate predictions
            batch_predictions = model.generate_prompt(embeddings)
            predictions.extend(batch_predictions)
            targets.extend(batch['prompt'])
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def train_epoch(model: EmbeddingToPromptModel, dataloader: DataLoader, 
               optimizer: torch.optim.Optimizer, scheduler: Any,
               device: torch.device, config: Dict[str, Any]) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Move to device
        embeddings = batch['embedding'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        if config['model']['architecture'] == 't5-small':
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            
            # Forward pass
            outputs = model(embeddings, decoder_input_ids, attention_mask)
            
            # Compute loss
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                input_ids.view(-1),
                ignore_index=model.tokenizer.pad_token_id
            )
            
        else:
            # For MLP and Transformer models
            outputs = model(embeddings, input_ids)
            
            # Compute loss
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                input_ids.view(-1),
                ignore_index=model.tokenizer.pad_token_id
            )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Log progress
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def save_checkpoint(model: EmbeddingToPromptModel, optimizer: torch.optim.Optimizer,
                   scheduler: Any, epoch: int, train_loss: float, val_loss: float,
                   save_path: str, config: Dict[str, Any]):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


def load_checkpoint(model: EmbeddingToPromptModel, optimizer: torch.optim.Optimizer,
                   scheduler: Any, checkpoint_path: str) -> Tuple[int, float, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train embedding-to-prompt model")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--data-path", help="Path to training data")
    parser.add_argument("--output-dir", default="models/checkpoints", help="Output directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate model")
    parser.add_argument("--debug", action="store_true", help="Debug mode with small dataset")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load environment variables
    load_dotenv()
    
    # Set random seeds
    random.seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    torch.manual_seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['training']['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine data path
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = Path(config['data']['embeddings_cache_path']) / 'training_data.jsonl'
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.error("Please run the embedding generation script first")
        return
    
    # Create data splits
    train_path, val_path, test_path = create_data_splits(data_path, config)
    
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    # Create datasets
    train_dataset = EmbeddingPromptDataset(
        train_path, tokenizer, config['model']['max_prompt_length'], 'train'
    )
    val_dataset = EmbeddingPromptDataset(
        val_path, tokenizer, config['model']['max_prompt_length'], 'val'
    )
    
    # Debug mode - use smaller dataset
    if args.debug:
        train_dataset.data = train_dataset.data[:100]
        val_dataset.data = val_dataset.data[:50]
        logger.info("Debug mode: Using smaller dataset")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    model = EmbeddingToPromptModel(config)
    model.to(device)
    
    logger.info(f"Model architecture: {config['model']['architecture']}")
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize scheduler
    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _, _ = load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch += 1
    
    # Evaluation only mode
    if args.eval_only:
        logger.info("Evaluation only mode")
        val_metrics = evaluate_model(model, val_loader, device, config)
        logger.info(f"Validation metrics: {val_metrics}")
        return
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, config)
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, device, config)
        val_loss = val_metrics['loss']
        
        logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        logger.info(f"Validation metrics: {val_metrics}")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, 
                       str(checkpoint_path), config)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = output_dir / "best_model.pt"
            model.save_pretrained(str(best_model_path))
            logger.info(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping after {patience} epochs without improvement")
            break
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main() 