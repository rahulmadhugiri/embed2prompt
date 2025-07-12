#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from typing import Dict, Any, Optional, List
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MLPEmbeddingToPrompt(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.embedding_dim = config['openai']['embedding_dimensions']
        self.hidden_size = config['model']['hidden_size']
        self.vocab_size = config['model'].get('vocab_size', 32128)
        self.max_prompt_length = config['model']['max_prompt_length']
        self.dropout = config['model']['dropout']
        
        self.input_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            for _ in range(config['model']['num_layers'])
        ])
        
        self.output_projection = nn.Linear(self.hidden_size, self.vocab_size * self.max_prompt_length)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(embedding)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_projection(x)
        return output.view(embedding.shape[0], self.max_prompt_length, self.vocab_size)


class TransformerEmbeddingToPrompt(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.embedding_dim = config['openai']['embedding_dimensions']
        self.hidden_size = config['model']['hidden_size']
        self.vocab_size = config['model'].get('vocab_size', 32128)
        self.max_prompt_length = config['model']['max_prompt_length']
        
        self.embedding_projection = nn.Linear(self.embedding_dim, self.hidden_size)
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.pos_encoding = PositionalEncoding(self.hidden_size, self.max_prompt_length)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=config['model']['num_heads'],
            dim_feedforward=self.hidden_size * 4,
            dropout=config['model']['dropout'],
            activation='relu',
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config['model']['num_layers'])
        self.output_projection = nn.Linear(self.hidden_size, self.vocab_size)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, embedding: torch.Tensor, target_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = embedding.shape[0]
        memory = self.embedding_projection(embedding).unsqueeze(1)
        
        if target_tokens is not None:
            seq_len = target_tokens.shape[1]
            token_emb = self.token_embedding(target_tokens)
            token_emb = self.pos_encoding(token_emb.transpose(0, 1)).transpose(0, 1)
            tgt_mask = self._generate_causal_mask(seq_len).to(embedding.device)
            output = self.transformer_decoder(tgt=token_emb, memory=memory, tgt_mask=tgt_mask)
        else:
            output = self._generate_autoregressive(memory, batch_size, embedding.device)
        
        return self.output_projection(output)
    
    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def _generate_autoregressive(self, memory: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        outputs = []
        
        for i in range(self.max_prompt_length):
            token_emb = self.token_embedding(generated)
            token_emb = self.pos_encoding(token_emb.transpose(0, 1)).transpose(0, 1)
            tgt_mask = self._generate_causal_mask(generated.shape[1]).to(device)
            
            output = self.transformer_decoder(tgt=token_emb, memory=memory, tgt_mask=tgt_mask)
            outputs.append(output)
            
            next_token_logits = self.output_projection(output[:, -1:, :])
            next_token = torch.argmax(next_token_logits, dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if torch.all(next_token == 1):  # EOS token
                break
        
        return torch.cat(outputs, dim=1)


class T5EmbeddingToPrompt(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.embedding_dim = config['openai']['embedding_dimensions']
        
        model_name = config['model'].get('pretrained_model', 't5-small')
        self.t5_config = T5Config.from_pretrained(model_name)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        self.embedding_projection = nn.Linear(self.embedding_dim, self.t5_config.d_model)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        if config['model'].get('freeze_encoder', False):
            for param in self.t5_model.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, embedding: torch.Tensor, target_tokens: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = embedding.shape[0]
        projected_embedding = self.embedding_projection(embedding)
        encoder_hidden_states = projected_embedding.unsqueeze(1)
        encoder_attention_mask = torch.ones(batch_size, 1, device=embedding.device)
        
        if target_tokens is not None:
            outputs = self.t5_model(
                encoder_outputs=(encoder_hidden_states,),
                attention_mask=encoder_attention_mask,
                decoder_input_ids=target_tokens,
                decoder_attention_mask=attention_mask
            )
            return outputs.logits
        else:
            return self.t5_model.generate(
                encoder_outputs=(encoder_hidden_states,),
                attention_mask=encoder_attention_mask,
                max_length=128,  # TODO: make configurable
                num_beams=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )


class EmbeddingToPromptModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.architecture = config['model']['architecture']
        
        if self.architecture == 'mlp':
            self.model = MLPEmbeddingToPrompt(config)
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        elif self.architecture == 'transformer':
            self.model = TransformerEmbeddingToPrompt(config)
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        elif self.architecture == 't5-small':
            self.model = T5EmbeddingToPrompt(config)
            self.tokenizer = self.model.tokenizer
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
    
    def forward(self, embedding: torch.Tensor, target_tokens: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.architecture == 't5-small':
            return self.model(embedding, target_tokens, attention_mask)
        else:
            return self.model(embedding, target_tokens)
    
    def generate_prompt(self, embedding: torch.Tensor, **kwargs) -> List[str]:
        self.eval()
        with torch.no_grad():
            if self.architecture == 't5-small':
                outputs = self.model(embedding)
                generated_ids = outputs.sequences
                return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            else:
                logits = self.model(embedding)
                generated_ids = torch.argmax(logits, dim=-1)
                return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_pretrained(self, save_path: str):
        import os
        os.makedirs(save_path, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, os.path.join(save_path, 'model.pt'))
        if hasattr(self, 'tokenizer'):
            self.tokenizer.save_pretrained(save_path)
    
    @classmethod
    def load_pretrained(cls, load_path: str):
        import os
        checkpoint = torch.load(os.path.join(load_path, 'model.pt'))
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 