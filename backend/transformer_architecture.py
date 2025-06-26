"""
Simple Vision Transformer for MNIST
==================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def _build_sincos_pos_embed(num_patches, embed_dim):
    """
    Build 1D sine-cosine positional encoding.
    Shape: [1, num_patches, embed_dim]
    """
    def get_angle(pos, i, d_model):
        return pos / (10000 ** (2 * (i // 2) / d_model))

    pos = torch.arange(num_patches).unsqueeze(1)  # [num_patches, 1]
    i = torch.arange(embed_dim).unsqueeze(0)      # [1, embed_dim]
    angle_rates = get_angle(pos, i, embed_dim)    # [num_patches, embed_dim]

    pos_encoding = torch.zeros_like(angle_rates)
    pos_encoding[:, 0::2] = torch.sin(angle_rates[:, 0::2])
    pos_encoding[:, 1::2] = torch.cos(angle_rates[:, 1::2])

    return pos_encoding.unsqueeze(0)  # [1, num_patches, embed_dim]

class VisionTransformer(nn.Module):

    def __init__(self, patch_dim=49, embed_dim=32, num_patches=16, num_classes=10, 
                 num_heads=4, num_layers=3, ffn_ratio=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 32 // 4 = 8 dimensions per head
        self.num_layers = num_layers
        
        # Patch embedding: converts flattened 7x7 patches (49 pixels) to embedding vectors
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)  # 49 -> 32
        
        # Learnable positional embeddings for each patch position
        self.register_buffer('pos_embedding', _build_sincos_pos_embed(num_patches, self.embed_dim)) # [1, 16, 32]
        
        # Multi-head attention components (we'll use these in the forward pass)
        self.W_q = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.W_k = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.W_v = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.W_o = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        
        # Feed-forward networks for each layer
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * ffn_ratio),  # 32 -> 64
                nn.ReLU(),
                nn.Linear(embed_dim * ffn_ratio, embed_dim)   # 64 -> 32
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization for each transformer layer (2 per layer: before attention, before FFN)
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # Final classification head
        self.classifier = nn.Linear(embed_dim, num_classes)  # 32 -> 10
        
    def forward(self, x):
        # INPUT: x shape = [batch_size, 16, 1, 7, 7] - 16 patches of 7x7 pixels each
        batch_size = x.shape[0]
        
        # Step 1: Flatten patches from [batch, 16, 1, 7, 7] to [batch, 16, 49]
        x = x.flatten(start_dim=2)  # Flatten the 1x7x7 = 49 pixels per patch
        # Data shape: [batch_size, 16, 49] - 16 patches, each with 49 pixel values
        
        # Step 2: Embed patches using linear projection [batch, 16, 49] -> [batch, 16, 32]
        x = self.patch_embedding(x)  # Convert 49-dim pixel vectors to 32-dim embeddings
        # Data shape: [batch_size, 16, 32] - 16 patch embeddings of 32 dimensions each
        
        # Step 3: Add positional embeddings to tell the model where each patch is located
        x = x + self.pos_embedding  # Broadcasting: [batch, 16, 32] + [1, 16, 32]
        # Data shape: [batch_size, 16, 32] - embeddings now contain positional information
        
        # Step 4: Pass through transformer layers
        for layer_idx in range(self.num_layers):
            # === MULTI-HEAD SELF-ATTENTION ===
            
            # Pre-attention layer norm
            residual = x  # Save for residual connection
            x = self.norm1_layers[layer_idx](x)  # Normalize before attention
            # Data shape: [batch_size, 16, 32] - normalized embeddings
            
            # Compute Query, Key, Value matrices
            Q = self.W_q[layer_idx](x)  # [batch, 16, 32] -> [batch, 16, 32]
            K = self.W_k[layer_idx](x)  # [batch, 16, 32] -> [batch, 16, 32]
            V = self.W_v[layer_idx](x)  # [batch, 16, 32] -> [batch, 16, 32]
            
            # Reshape for multi-head attention: split embedding into num_heads
            # From [batch, 16, 32] to [batch, 4, 16, 8] - 4 heads, each with 8 dimensions
            Q = Q.view(batch_size, 16, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, 16, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, 16, self.num_heads, self.head_dim).transpose(1, 2)
            # Data shape: [batch_size, 4, 16, 8] - 4 attention heads, 16 patches, 8 dims per head
            
            # Scaled dot-product attention
            # Q @ K^T gives attention scores between all pairs of patches
            attention_scores = Q @ K.transpose(-2, -1)  # [batch, 4, 16, 16]
            attention_scores = attention_scores / (self.head_dim ** 0.5)  # Scale by sqrt(head_dim)
            # Data shape: [batch_size, 4, 16, 16] - attention scores between all patch pairs
            
            # Softmax to get attention weights (probabilities)
            attention_weights = F.softmax(attention_scores, dim=-1)
            # Data shape: [batch_size, 4, 16, 16] - normalized attention weights
            
            # Apply attention weights to values
            attention_output = attention_weights @ V  # [batch, 4, 16, 16] @ [batch, 4, 16, 8]
            # Data shape: [batch_size, 4, 16, 8] - attended values for each head
            
            # Concatenate heads: [batch, 4, 16, 8] -> [batch, 16, 32]
            attention_output = attention_output.transpose(1, 2).contiguous()
            attention_output = attention_output.view(batch_size, 16, self.embed_dim)
            # Data shape: [batch_size, 16, 32] - concatenated multi-head attention output
            
            # Final linear projection of attention output
            x = self.W_o[layer_idx](attention_output)
            # Data shape: [batch_size, 16, 32] - projected attention output
            
            # Residual connection: add input to attention output
            x = residual + x
            # Data shape: [batch_size, 16, 32] - with residual connection
            
            # === FEED-FORWARD NETWORK ===
            
            # Pre-FFN layer norm
            residual = x  # Save for residual connection
            x = self.norm2_layers[layer_idx](x)  # Normalize before FFN
            # Data shape: [batch_size, 16, 32] - normalized embeddings
            
            # Feed-forward network: 32 -> 64 -> 32 (with ReLU in between)
            x = self.ffn_layers[layer_idx](x)
            # Data shape: [batch_size, 16, 32] - FFN output
            
            # Residual connection: add input to FFN output
            x = residual + x
            # Data shape: [batch_size, 16, 32] - final layer output with residual
        
        # Step 5: Global average pooling - average across all patches
        x = x.mean(dim=1)  # Average over the 16 patches dimension
        # Data shape: [batch_size, 32] - single embedding vector per image
        
        # Step 6: Classification head - convert to class probabilities
        logits = self.classifier(x)  # [batch, 32] -> [batch, 10]
        # Data shape: [batch_size, 10] - logits for 10 digit classes (0-9)
        
        return logits 



class DigitDecoder(nn.Module):
    def __init__(self, d_model=128, num_layers=2, num_heads=4, max_len=4):
        super().__init__()
        self.token_embedding = nn.Embedding(11, d_model)  # 0-9 digits + START
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=512)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output_proj = nn.Linear(d_model, 10)  # output: 0-9 digits only

    def forward(self, tgt_tokens, encoder_output):
        # tgt_tokens: [batch, seq_len] - input digits (shifted right)
        # encoder_output: [batch, src_len, d_model]

        tgt_emb = self.token_embedding(tgt_tokens) + self.pos_embedding[:, :tgt_tokens.size(1), :]
        tgt_emb = tgt_emb.transpose(0, 1)  # [seq_len, batch, d_model]
        memory = encoder_output.transpose(0, 1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.size(1)).to(tgt_tokens.device)

        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)  # [seq_len, batch, d_model]
        logits = self.output_proj(out.transpose(0, 1))  # [batch, seq_len, 10]
        return logits
