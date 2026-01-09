from torch import nn
import numpy as np
import torch

from model.positional_encoding import PositionalEncoding


class TrunkNetModule(nn.Module):
    """ Trunk network based on a transformer architecture """
    
    def __init__(
        self,
        embedding_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.0,
        mha_layers: int = 2,
        hidden_dims: list[int] = None
    ):
        """
        
        """
        super(TrunkNetModule, self).__init__()

        self.mha_layers = mha_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # (batch_size, 100, 4)
        # -> batch_size elements in batch, sequence of 100 one-hot encoded entries, 4 encoding dimensions
        # -> up-project encodings to embeddings for more self-attention
        self.linear_embedding = nn.Linear(4, self.embedding_dim, bias = True)
        
        # Create positional encodings
        self.positional_encoding = PositionalEncoding(self.embedding_dim)#, self.dropout)

        # (batch_size, 100, embedding_dim)
        # Multi-headed (self-)attention
        self.mha = nn.ModuleList([
            nn.MultiheadAttention(
                    self.embedding_dim,
                    self.num_heads,
                    bias = True,
                    dropout = self.dropout,
                    batch_first = True
            )
            for _ in range(self.mha_layers)
        ])

        # Layer normalization within MHA
        self.mha_lnorm = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim)
            for _ in range(self.mha_layers)
        ])

        # Linear layer within MHA
        self.mha_linear = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias = True),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            for _ in range(self.mha_layers - 1)
        ])
        
        # Layer normalization within MHA
        self.mha_linear_lnorm = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim)
            for _ in range(self.mha_layers - 1)
        ])
        
        self.seq_compress = nn.Sequential(
            nn.Linear(100, 1, bias = True),
            nn.Tanh(),
            nn.Dropout(self.dropout)
        )

        # Linear layers at the end
        self.linear = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[idx], hidden_dims[idx + 1], bias = True),
                nn.LayerNorm(hidden_dims[idx + 1]),
                nn.Dropout(self.dropout),
                nn.ReLU() if idx + 1 < len(hidden_dims) - 1 else nn.Tanh(),
            )
            for idx in range(len(hidden_dims) - 1)
        ])
    
    
    def forward(
        self,
        data
    ):
        """
        
        """
        # might not need attention output weights _att1ow
        
        _out = data
        
        # Embed input and add positional encodings
        _out = self.linear_embedding(_out) * np.sqrt(self.embedding_dim)
        _out = self.positional_encoding(_out)

        _attn_maps = []
        for mha_layer in range(self.mha_layers):
            # Apply self-attention
            _atto, _attow = self.mha[mha_layer](
                query = _out, key = _out, value = _out,
                need_weights = True, average_attn_weights = False
            )
            _attn_maps.append(_attow)
            
            # Residual connection and layer normalization
            _out = _out + _atto
            _out = self.mha_lnorm[mha_layer](_out)
            
            # Linear layer
            if mha_layer != self.mha_layers - 1:
                # Apply linear transform
                _lino = self.mha_linear[mha_layer](_out)
                
                # Residual connection and layer normalization
                _out = _out + _lino
                _out = self.mha_linear_lnorm[mha_layer](_out)
        
        # Compress sequence dimension
        _out = _out.transpose(1, 2)     # New shape: (batch_size, hidden_dim, seq_len)
        _out = self.seq_compress(_out)  # New shape: (batch_size, hidden_dim, 1)
        _out = _out.squeeze(-1)         # Final shape: (batch_size, hidden_dim)
        
        for hidden_layer in self.linear:
            _out = hidden_layer(_out)
        
        # Make list into a tensor of shape (batch_size, mha_layers, num_heads, 100, 100)
        _attn_maps = torch.stack(_attn_maps, dim = 1)
        
        return _out, _attn_maps