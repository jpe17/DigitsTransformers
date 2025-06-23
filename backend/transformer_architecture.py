"""
Simple Vision Transformer for MNIST
==================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, ffn_ratio=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Standard multi-head attention (split embedding)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_ratio),
            nn.ReLU(),
            nn.Linear(embed_dim * ffn_ratio, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Multi-head self-attention
        residual = x
        x = self.norm1(x)
        
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Compute Q, K, V and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = attention_weights @ V
        
        # Concatenate heads and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        x = self.W_o(attention_output)
        x = residual + x  # Residual connection
        
        # Feed-forward network
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x  # Residual connection
        
        return x


class VisionTransformerEncoder(nn.Module):
    def __init__(self, patch_dim=49, embed_dim=32, num_patches=16, num_classes=10, 
                 num_heads=4, num_layers=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Patch embedding and positional encoding
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        # Transformer layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # print(f"\n1. Input patches: {x.shape}")
        
        # Flatten patches: [batch, 16, 1, 7, 7] -> [batch, 16, 49]
        x = x.flatten(start_dim=2)
        # print(f"2. Flattened: {x.shape}")
        
        # Embed patches: [batch, 16, 49] -> [batch, 16, 32]
        x = self.patch_embedding(x)
        # print(f"3. Embedded: {x.shape}")
        
        # Add positional embeddings
        x = x + self.pos_embedding
        # print(f"4. With positions: {x.shape}")
        
        # Pass through transformer layers
        # print(f"5. Through {self.num_layers} transformer layers:")
        for i, layer in enumerate(self.encoder_layers):
            # print(f"  Layer {i+1}:")
            x = layer(x)
        
        # Global average pooling and classification
        x = x.mean(dim=1)  # [batch, 32]
        # print(f"6. After pooling: {x.shape}")
        
        logits = self.classifier(x)  # [batch, 10]
        # print(f"7. Final logits: {logits.shape}")
        
        return logits 