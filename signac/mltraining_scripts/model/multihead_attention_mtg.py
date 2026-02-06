"""
This module was implemented by DeepSeek then validated experimentally for implementing the
'Mind the Gap' paper fix.

Soon should follow stylistic uniformization with the other modules
"""

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

import pandas as pd

from mltraining_scripts.model.positional_encoding import RelativePositionEncoding


class MultiheadAttentionMTG(nn.Module):
	"""
	Modified MultiheadAttention with optional 'Mind the Gap' fix applied post-softmax.
	When mind_the_gap=False, should produce identical outputs to nn.MultiheadAttention
	given same initialization and inputs.
	"""
	
	def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, 
				 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
				 batch_first=False, device=None, dtype=None):
		super().__init__()
		factory_kwargs = {'device': device, 'dtype': dtype}
		
		# Store critical parameters
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.dropout = dropout
		self.batch_first = batch_first
		self.head_dim = embed_dim // num_heads
		assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
		
		self.rel_pos = RelativePositionEncoding(embed_dim, scaling_type='linear', k = -1/np.sqrt(embed_dim))
		
		# Replicate PyTorch's internal projections
		self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
		self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
		self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
		self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
		
		# Store dropout module
		self.dropout_module = nn.Dropout(dropout)
	
	
	def forward(self, query, key, value, key_padding_mask=None, need_weights=True, 
				attn_mask=None, average_attn_weights=True, temperature=1.0,
				mind_the_gap: bool = True):
		"""
		Forward pass with optional 'Mind the Gap' fix.
		
		Args:
			temperature: Scaling factor for Q before attention
			mind_the_gap: If True, apply the uniform component subtraction fix
		"""
		# Handle batch_first format
		if self.batch_first:
			query = query.transpose(0, 1)
			key = key.transpose(0, 1)
			value = value.transpose(0, 1)
		
		tgt_len, bsz, embed_dim = query.shape
		src_len = key.size(0)
		
		# Apply projections
		q = self.q_proj(query)  # [tgt_len, bsz, embed_dim]
		k = self.k_proj(key)    # [src_len, bsz, embed_dim]
		v = self.v_proj(value)  # [src_len, bsz, embed_dim]
		
		# Apply temperature scaling
		q = q / temperature
		
		# Reshape for multi-head computation
		q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
		k = k.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
		v = v.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
		
		# cos_sim = F.cosine_similarity(q.flatten(2), k.flatten(2), dim=-1)
		# summary = pd.Series(cos_sim.flatten().detach().numpy()).describe().values
		# print(f"Q-K cosine similarity: {summary[1:3]} | {summary[3:]}")
		
		# Compute attention scores
		attn_output_weights = torch.bmm(q, k.transpose(1, 2))  # [bsz*heads, tgt_len, src_len]
		attn_output_weights /= (self.head_dim ** 0.5)
		
		# Add relative positional embeddings
		attn_output_weights = self.rel_pos(attn_output_weights)
		
		# Apply attention mask if provided
		if attn_mask is not None:
			attn_output_weights += attn_mask
		
		# Apply key padding mask if provided
		if key_padding_mask is not None:
			attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
			attn_output_weights = attn_output_weights.masked_fill(
				key_padding_mask.unsqueeze(1).unsqueeze(2),
				float('-inf')
			)
			attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)
		
		# Apply softmax
		attn_weights_raw = F.softmax(attn_output_weights, dim=-1)
		
		# --- CONDITIONAL: 'Mind the Gap' fix ---
		if mind_the_gap:
			# Subtract the row-wise mean (uniform component)
			uniform_component = attn_weights_raw.mean(dim=-1, keepdim=True)  # [bsz*heads, tgt_len, 1]
			attn_weights = attn_weights_raw - uniform_component
		else:
			# Keep original attention weights (matching PyTorch's behavior)
			attn_weights = attn_weights_raw
		
		# Apply dropout to the attention weights
		attn_weights = self.dropout_module(attn_weights)
		
		# Apply attention weights to values
		attn_output = torch.bmm(attn_weights, v)  # [bsz*heads, tgt_len, head_dim]
		
		# Reshape back
		attn_output = attn_output.transpose(0, 1).reshape(tgt_len, bsz, embed_dim)
		
		# Apply output projection
		attn_output = self.out_proj(attn_output)
		
		# Handle batch_first format
		if self.batch_first:
			attn_output = attn_output.transpose(0, 1)
		
		# Prepare return values
		if need_weights:
			# Reshape weights for return
			attn_weights_return = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
			if average_attn_weights:
				attn_weights_return = attn_weights_return.mean(dim=1)
			return attn_output, attn_weights_return
		else:
			return attn_output