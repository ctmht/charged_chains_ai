from torch import nn
import numpy as np
import torch

from mltraining_scripts.model.positional_encoding import PositionalEncoding
from mltraining_scripts.model.multihead_attention_mtg import MultiheadAttentionMTG


class TrunkNetModule(nn.Module):
    """ Trunk network based on a transformer architecture """
    
    def __init__(
        self,
        embedding_dim: int = 32,
        num_heads: int = 4,
        mha_layers: int = 2,
        hidden_dims_within: list[int] = None,
        hidden_dims_after: list[int] = None,
        dropout: float = 0.0,
        temperature: float = 1.0
    ):
        """
        
        """
        if hidden_dims_within is None or len(hidden_dims_within) != mha_layers:
            raise ValueError(
                f"Hidden dimensions for feed-forward transformer modules must be provided for "
                f"each of the {mha_layers} layers, got {hidden_dims_within} with "
                f"{0 if hidden_dims_within is None else len(hidden_dims_within)} entries"
            )
            
        
        super(TrunkNetModule, self).__init__()

        self.mha_layers = mha_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.temperature = temperature
        
        # (batch_size, 100, 4)
        # -> batch_size elements in batch, sequence of 100 one-hot encoded entries, 4 encoding dimensions
        # -> up-project encodings to embeddings for more self-attention
        self.linear_embedding = nn.Sequential(
            nn.Linear(4, self.embedding_dim, bias = True),
            # nn.Tanh()
        )
        
        # Create positional encodings (no dropout)
        self.positional_encoding = PositionalEncoding(self.embedding_dim)
        
        # Layer normalization within MHA
        # Pre-LN
        self.mha_lnorm_pre = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim)
            for _ in range(self.mha_layers)
        ])
        
        # (batch_size, 100, embedding_dim)
        # Multi-headed (self-)attention
        self.mha = nn.ModuleList([
            # MultiheadAttentionMTG(
            nn.MultiheadAttention(
                self.embedding_dim,
                self.num_heads,
                bias = False,
                dropout = self.dropout,
                batch_first = True
            )
            for _ in range(self.mha_layers)
        ])
        
        # Layer normalization within MHA
        # Pre-LN
        self.mha_linear_lnorm_pre = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim)
            for _ in range(self.mha_layers)
        ])

        # Linear layer within MHA
        self.mha_feedforward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim, hidden_dims_within[idx], bias = True),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_dims_within[idx], self.embedding_dim, bias = True),
                nn.Dropout(self.dropout),
            )
            for idx in range(self.mha_layers)
        ])
        
        # Linear layers at the end
        self.linear = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims_after[idx], hidden_dims_after[idx + 1], bias = True),
                nn.LayerNorm(hidden_dims_after[idx + 1]),
                nn.Dropout(self.dropout),
                nn.ReLU() #if idx + 1 < len(hidden_dims_after) - 1 else nn.Tanh(),
            )
            for idx in range(len(hidden_dims_after) - 1)
        ])
        
        # # Compression of the embedding dimension
        # self.embdim_compress = nn.Sequential(
        #     nn.Linear(self.embedding_dim, 1, bias = True),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout)
        # )
    
    
    def forward(
        self,
        data
    ):
        """
        
        """
        # might not need attention output weights _att1ow
        
        _out = data
        
        # Embed input and add positional encodings
        _out = self.linear_embedding(_out) #* np.sqrt(self.embedding_dim)
        _out = self.positional_encoding(_out)

        _attn_maps = []
        for mha_layer in range(self.mha_layers):
            # Pre-LN
            _lnormed_mha = self.mha_lnorm_pre[mha_layer](_out)
            
            # Self-attention
            _atto, _attow = self.mha[mha_layer](
                query = _lnormed_mha / self.temperature,
                key = _lnormed_mha,
                value = _lnormed_mha,
                need_weights = True,
                average_attn_weights = False
            )
            _attn_maps.append(_attow)
            
            # Residual connection
            _out = _out + _atto
            
            # Pre-LN
            _lnormed_mha_linear = self.mha_linear_lnorm_pre[mha_layer](_out)
            
            # Linear layer
            _lino = self.mha_feedforward[mha_layer](_lnormed_mha_linear)
            
            # Residual connection
            _out = _out + _lino
        
        
        for hidden_layer in self.linear:
            _out = hidden_layer(_out)
        
        # Compress sequence dimension
        # _out = _out.transpose(1, 2)     # New shape: (batch_size, hidden_dim, seq_len)
        # _out = self.seq_compress(_out)  # New shape: (batch_size, hidden_dim, 1)
        # _out = _out.squeeze(-1)         # Final shape: (batch_size, hidden_dim)
        
        _seq_meanpooled = _out.mean(dim = 1)
        
        # Make list into a tensor of shape (batch_size, mha_layers, num_heads, 100, 100)
        _attn_maps = torch.stack(_attn_maps, dim = 1)
        
        return _seq_meanpooled, _attn_maps