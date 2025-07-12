#!/usr/bin/env python3
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

import sys
sys.path.append('.')
from models.architecture import EmbeddingToPromptModel
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingPromptDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: T5Tokenizer, max_length: int, split: str = 'train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        self.data = []
        try:
            with jsonlines.open(data_path, 'r') as reader:
                for item in reader:
                    if 'embedding' in item and 'prompt' in item:
                        self.data.append(item)
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            raise
        
        logger.info(f"Loaded {len(self.data)} samples for {split}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        embedding = np.array(item['embedding'], dtype=np.float32)
        prompt = item['prompt']
        
        # Tokenize prompt
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'embedding': torch.tensor(embedding),
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }
        
        # Add decoder input ids for T5
        if self.tokenizer.model_max_length:
            decoder_input_ids = self.tokenizer.prepare_seq2seq_batch(
                tgt_texts=[prompt],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )['input_ids'].squeeze()
            result['decoder_input_ids'] = decoder_input_ids
        
        return result


def create_data_splits(data_path: str, config: Dict[str, Any]) -> Tuple[str, str, str]:
    logger.info(f"Creating data splits from {data_path}")
    
    # Load all data
    data = []
    with jsonlines.open(data_path, 'r') as reader:
        data = list(reader)
    
    # Shuffle data
    random.shuffle(data)
    
    # Split data
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    
    n_train = int(len(data) * train_split)
    n_val = int(len(data) * val_split)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    # Save splits
    data_dir = Path(data_path).parent
    train_path = data_dir / 'train.jsonl'
    val_path = data_dir / 'val.jsonl'
    test_path = data_dir / 'test.jsonl'
    
    for split_data, split_path in [(train_data, train_path), (val_data, val_path), (test_data, test_path)]:
        with jsonlines.open(split_path, 'w') as writer:
            for item in split_data:
                writer.write(item)
    
    logger.info(f"Created splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return str(train_path), str(val_path), str(test_path)


def train_epoch(model: EmbeddingToPromptModel, dataloader: DataLoader, 
               optimizer: torch.optim.Optimizer, scheduler: Any,
               device: torch.device, config: Dict[str, Any]) -> float:
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        embeddings = batch['embedding'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        if config['model']['architecture'] == 't5-small':
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            outputs = model(embeddings, decoder_input_ids, attention_mask)
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                input_ids.view(-1),
                ignore_index=model.tokenizer.pad_token_id
            )
        else:
            outputs = model(embeddings, input_ids)
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                input_ids.view(-1),
                ignore_index=model.tokenizer.pad_token_id
            )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def evaluate_model(model: EmbeddingToPromptModel, dataloader: DataLoader, 
                  device: torch.device, config: Dict[str, Any]) -> float:
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeddings = batch['embedding'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            if config['model']['architecture'] == 't5-small':
                decoder_input_ids = batch['decoder_input_ids'].to(device)
                outputs = model(embeddings, decoder_input_ids, attention_mask)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    input_ids.view(-1),
                    ignore_index=model.tokenizer.pad_token_id
                )
            else:
                outputs = model(embeddings, input_ids)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    input_ids.view(-1),
                    ignore_index=model.tokenizer.pad_token_id
                )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def save_checkpoint(model: EmbeddingToPromptModel, optimizer: torch.optim.Optimizer,
                   scheduler: Any, epoch: int, loss: float, save_path: str):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': model.config
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(model: EmbeddingToPromptModel, optimizer: torch.optim.Optimizer,
                   scheduler: Any, checkpoint_path: str) -> Tuple[int, float, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    config = checkpoint.get('config', {})
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}, loss {loss:.4f}")
    return epoch, loss, config


def main():
    parser = argparse.ArgumentParser(description="Train embedding-to-prompt model")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--data-path", help="Path to training data")
    parser.add_argument("--output-dir", default="models/checkpoints", help="Output directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate model")
    parser.add_argument("--debug", action="store_true", help="Debug mode with small dataset")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    load_dotenv()
    
    # Set random seeds
    random.seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    torch.manual_seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['training']['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = Path(config['data']['embeddings_cache_path']) / 'training_data.jsonl'
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    train_path, val_path, test_path = create_data_splits(data_path, config)
    
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    train_dataset = EmbeddingPromptDataset(train_path, tokenizer, config['model']['max_prompt_length'], 'train')
    val_dataset = EmbeddingPromptDataset(val_path, tokenizer, config['model']['max_prompt_length'], 'val')
    
    if args.debug:
        train_dataset.data = train_dataset.data[:100]
        val_dataset.data = val_dataset.data[:50]
        logger.info("Debug mode: Using smaller dataset")
    
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
    
    model = EmbeddingToPromptModel(config)
    model.to(device)
    
    logger.info(f"Model architecture: {config['model']['architecture']}")
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=total_steps
    )
    
    start_epoch = 0
    if args.resume:
        start_epoch, _, _ = load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch += 1
    
    if args.eval_only:
        val_loss = evaluate_model(model, val_loader, device, config)
        logger.info(f"Validation loss: {val_loss:.4f}")
        return
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, config)
        val_loss = evaluate_model(model, val_loader, device, config)
        
        logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, str(checkpoint_path))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / "best_model.pt"
            model.save_pretrained(str(best_model_path).replace('.pt', ''))
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main() 