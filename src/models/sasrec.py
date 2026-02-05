"""
SASRec (Self-Attentive Sequential Recommendation) model implementation.

Paper: Self-Attentive Sequential Recommendation (ICDM 2018)
https://arxiv.org/abs/1808.09781

This implementation follows RecBole's SASRec closely for reproducibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation model.
    
    Architecture matches RecBole implementation for reproducibility.
    
    Args:
        num_items: Total number of items (vocab size, index 0 = padding)
        d_model: Hidden dimension (default: 64, RecBole default)
        nhead: Number of attention heads (default: 2)
        num_layers: Number of Transformer blocks (default: 2)
        dropout: Dropout probability (default: 0.2)
        maxlen: Maximum sequence length (default: 50)
    """
    
    def __init__(
        self,
        num_items: int,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        maxlen: int = 50,
    ):
        super().__init__()
        
        self.num_items = num_items
        self.d_model = d_model
        self.maxlen = maxlen
        
        # Item embedding (index 0 = padding, will be masked)
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=d_model,
            padding_idx=0,
        )
        
        # Learnable positional embedding
        self.position_embedding = nn.Embedding(
            num_embeddings=maxlen,
            embedding_dim=d_model,
        )
        
        # Embedding dropout
        self.emb_dropout = nn.Dropout(dropout)
        
        # Layer normalization (applied before transformer)
        self.emb_layernorm = nn.LayerNorm(d_model, eps=1e-8)
        
        # Transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # Post-norm like original SASRec
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model, eps=1e-8),
        )
        
        # Output projection: hidden -> item scores
        # Using tied embeddings (shared with item_embedding)
        self.output_bias = nn.Parameter(torch.zeros(num_items))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following BERT/SASRec conventions."""
        # Item embedding
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        # Zero out padding embedding
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0)
        
        # Position embedding
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask (upper triangular = -inf).
        
        Prevents attending to future positions.
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Item ID sequence [batch_size, seq_len], 0 = padding
            
        Returns:
            Logits over all items [batch_size, seq_len, num_items]
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed items and positions
        item_emb = self.item_embedding(x)           # [B, L, D]
        pos_emb = self.position_embedding(positions) # [B, L, D]
        
        # Combine and normalize
        hidden = item_emb + pos_emb
        hidden = self.emb_layernorm(hidden)
        hidden = self.emb_dropout(hidden)
        
        # Causal mask for autoregressive attention
        causal_mask = self._create_causal_mask(seq_len, device)
        
        # Padding mask: True where input is padding (0)
        padding_mask = (x == 0)  # [B, L]
        
        # Apply transformer
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )  # [B, L, D]
        
        # Project to item space using tied embeddings
        # logits = hidden @ W^T + b, where W = item_embedding.weight
        logits = F.linear(hidden, self.item_embedding.weight, self.output_bias)
        # Shape: [B, L, num_items]
        
        return logits
    
    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for the next item (last position only).
        
        Args:
            x: Item ID sequence [batch_size, seq_len]
            
        Returns:
            Scores over items [batch_size, num_items]
        """
        logits = self.forward(x)  # [B, L, V]
        return logits[:, -1, :]   # [B, V] - last position


if __name__ == "__main__":
    # Quick test
    print("Testing SASRec model...")
    
    num_items = 3670
    model = SASRec(num_items=num_items, d_model=64, nhead=2, num_layers=2, maxlen=50)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 49  # maxlen - 1 (after splitting input/target)
    
    # Simulate input with padding on left
    x = torch.zeros(batch_size, seq_len, dtype=torch.long)
    x[:, -10:] = torch.randint(1, num_items, (batch_size, 10))  # Last 10 items are real
    
    print(f"\nInput shape: {x.shape}")
    print(f"Sample input: {x[0].tolist()}")
    
    logits = model(x)
    print(f"Output shape: {logits.shape}")  # Should be [4, 49, 3670]
    
    # Check that padding positions don't produce crazy values
    last_logits = model.predict_next(x)
    print(f"Last position logits shape: {last_logits.shape}")  # [4, 3670]
    print(f"Logits range: [{last_logits.min().item():.2f}, {last_logits.max().item():.2f}]")
