#!/usr/bin/env python3
"""
Model Architecture for Embedding-to-Prompt Generation
Supports both MLP and T5-based encoder-decoder architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MLPEmbeddingToPrompt(nn.Module):
    """
    Simple MLP-based model for embedding-to-prompt generation.
    Takes embedding vector and generates prompt tokens.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.embedding_dim = config['openai']['embedding_dimensions']
        self.hidden_size = config['model']['hidden_size']
        self.vocab_size = config['model'].get('vocab_size', 32128)  # T5 vocab size
        self.max_prompt_length = config['model']['max_prompt_length']
        self.dropout = config['model']['dropout']
        
        # Embedding projection layers
        self.input_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            for _ in range(config['model']['num_layers'])
        ])
        
        # Output projection to generate token logits
        self.output_projection = nn.Linear(self.hidden_size, self.vocab_size * self.max_prompt_length)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            embedding: Input embedding tensor of shape (batch_size, embedding_dim)
            
        Returns:
            logits: Token logits of shape (batch_size, max_prompt_length, vocab_size)
        """
        # Project embedding to hidden size
        x = self.input_projection(embedding)
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Project to output space
        output = self.output_projection(x)
        
        # Reshape to (batch_size, max_prompt_length, vocab_size)
        batch_size = embedding.shape[0]
        logits = output.view(batch_size, self.max_prompt_length, self.vocab_size)
        
        return logits


class TransformerEmbeddingToPrompt(nn.Module):
    """
    Transformer-based model for embedding-to-prompt generation.
    Uses a decoder-only architecture with embedding conditioning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.embedding_dim = config['openai']['embedding_dimensions']
        self.hidden_size = config['model']['hidden_size']
        self.vocab_size = config['model'].get('vocab_size', 32128)
        self.max_prompt_length = config['model']['max_prompt_length']
        self.num_heads = config['model']['num_heads']
        self.num_layers = config['model']['num_layers']
        self.dropout = config['model']['dropout']
        
        # Embedding projection
        self.embedding_projection = nn.Linear(self.embedding_dim, self.hidden_size)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.hidden_size, self.max_prompt_length)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=self.num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, self.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, embedding: torch.Tensor, target_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            embedding: Input embedding tensor of shape (batch_size, embedding_dim)
            target_tokens: Target token ids for teacher forcing (batch_size, seq_len)
            
        Returns:
            logits: Token logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size = embedding.shape[0]
        
        # Project embedding to hidden size and add batch dimension
        memory = self.embedding_projection(embedding).unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        if target_tokens is not None:
            # Teacher forcing mode
            seq_len = target_tokens.shape[1]
            
            # Token embeddings
            token_emb = self.token_embedding(target_tokens)  # (batch_size, seq_len, hidden_size)
            
            # Add positional encoding
            token_emb = self.pos_encoding(token_emb.transpose(0, 1)).transpose(0, 1)
            
            # Create causal mask
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(embedding.device)
            
            # Transformer decoder
            output = self.transformer_decoder(
                tgt=token_emb,
                memory=memory,
                tgt_mask=tgt_mask
            )
            
        else:
            # Inference mode - generate tokens autoregressively
            output = self._generate_autoregressive(memory, batch_size, embedding.device)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square causal mask for the sequence."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _generate_autoregressive(self, memory: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate tokens autoregressively."""
        # Start with BOS token (assuming token id 0)
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        outputs = []
        
        for i in range(self.max_prompt_length):
            # Token embeddings
            token_emb = self.token_embedding(generated)
            
            # Add positional encoding
            token_emb = self.pos_encoding(token_emb.transpose(0, 1)).transpose(0, 1)
            
            # Create causal mask
            tgt_mask = self._generate_square_subsequent_mask(generated.shape[1]).to(device)
            
            # Transformer decoder
            output = self.transformer_decoder(
                tgt=token_emb,
                memory=memory,
                tgt_mask=tgt_mask
            )
            
            outputs.append(output)
            
            # Get next token (greedy decoding)
            next_token_logits = self.output_projection(output[:, -1:, :])
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have generated EOS token (assuming token id 1)
            if torch.all(next_token == 1):
                break
        
        return torch.cat(outputs, dim=1)


class T5EmbeddingToPrompt(nn.Module):
    """
    T5-based model for embedding-to-prompt generation.
    Uses pre-trained T5 with custom embedding conditioning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.embedding_dim = config['openai']['embedding_dimensions']
        
        # Load pre-trained T5 model
        model_name = config['model'].get('pretrained_model', 't5-small')
        self.t5_config = T5Config.from_pretrained(model_name)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Embedding projection to T5's hidden size
        self.embedding_projection = nn.Linear(
            self.embedding_dim, 
            self.t5_config.d_model
        )
        
        # Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Freeze T5 encoder if specified
        if config['model'].get('freeze_encoder', False):
            for param in self.t5_model.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, embedding: torch.Tensor, target_tokens: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through T5 with embedding conditioning.
        
        Args:
            embedding: Input embedding tensor of shape (batch_size, embedding_dim)
            target_tokens: Target token ids for teacher forcing
            attention_mask: Attention mask for target tokens
            
        Returns:
            logits: Token logits from T5 decoder
        """
        batch_size = embedding.shape[0]
        
        # Project embedding to T5's hidden size
        projected_embedding = self.embedding_projection(embedding)
        
        # Create encoder inputs (single token per embedding)
        encoder_hidden_states = projected_embedding.unsqueeze(1)  # (batch_size, 1, d_model)
        encoder_attention_mask = torch.ones(batch_size, 1, device=embedding.device)
        
        if target_tokens is not None:
            # Training mode with teacher forcing
            outputs = self.t5_model(
                encoder_outputs=(encoder_hidden_states,),
                attention_mask=encoder_attention_mask,
                decoder_input_ids=target_tokens,
                decoder_attention_mask=attention_mask
            )
            return outputs.logits
        else:
            # Inference mode
            outputs = self.t5_model.generate(
                encoder_outputs=(encoder_hidden_states,),
                attention_mask=encoder_attention_mask,
                max_length=self.config['model']['max_prompt_length'],
                num_beams=self.config['inference'].get('num_beams', 1),
                do_sample=True,
                temperature=self.config['inference'].get('temperature', 0.7),
                top_p=self.config['inference'].get('top_p', 0.9),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            return outputs


class EmbeddingToPromptModel(nn.Module):
    """
    Main model class that wraps different architectures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.architecture = config['model']['architecture']
        
        # Initialize the appropriate model
        if self.architecture == 'mlp':
            self.model = MLPEmbeddingToPrompt(config)
        elif self.architecture == 'transformer':
            self.model = TransformerEmbeddingToPrompt(config)
        elif self.architecture == 't5-small':
            self.model = T5EmbeddingToPrompt(config)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Initialize tokenizer for text generation
        if self.architecture != 't5-small':
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        else:
            self.tokenizer = self.model.tokenizer
    
    def forward(self, embedding: torch.Tensor, target_tokens: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model."""
        
        if self.architecture == 't5-small':
            return self.model(embedding, target_tokens, attention_mask)
        else:
            return self.model(embedding, target_tokens)
    
    def generate_prompt(self, embedding: torch.Tensor, **kwargs) -> List[str]:
        """
        Generate prompt text from embedding.
        
        Args:
            embedding: Input embedding tensor
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated prompt strings
        """
        self.eval()
        with torch.no_grad():
            if self.architecture == 't5-small':
                # T5 has built-in generation
                outputs = self.model(embedding)
                generated_ids = outputs.sequences
                
                # Decode generated tokens
                prompts = []
                for ids in generated_ids:
                    prompt = self.tokenizer.decode(ids, skip_special_tokens=True)
                    prompts.append(prompt)
                
                return prompts
            
            else:
                # For MLP and Transformer, use simple decoding
                logits = self.model(embedding)
                
                # Greedy decoding
                generated_ids = torch.argmax(logits, dim=-1)
                
                # Decode generated tokens
                prompts = []
                for ids in generated_ids:
                    prompt = self.tokenizer.decode(ids, skip_special_tokens=True)
                    prompts.append(prompt)
                
                return prompts
    
    def get_num_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_pretrained(self, save_path: str):
        """Save the model."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, os.path.join(save_path, 'model.pt'))
        
        # Save tokenizer
        if hasattr(self, 'tokenizer'):
            self.tokenizer.save_pretrained(save_path)
    
    @classmethod
    def load_pretrained(cls, load_path: str):
        """Load a pre-trained model."""
        import os
        
        # Load model checkpoint
        checkpoint = torch.load(os.path.join(load_path, 'model.pt'))
        config = checkpoint['config']
        
        # Create model
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model 