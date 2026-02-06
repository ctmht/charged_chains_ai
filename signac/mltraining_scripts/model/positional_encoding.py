from torch import nn
import numpy as np
import torch

from matplotlib import pyplot as plt
import seaborn as sns


class PositionalEncoding(nn.Module):
	""" Sinusoidal positional encodings for transformer """
	"""
	Sourced from PyTorch tutorial 'Language Modeling with nn.Transformer and torchtext'
	URL: https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
	"""
	
	def __init__(
		self,
		d_model: int,
		dropout: float = 0.0,
		max_len: int = 100
	):
		"""
		
		"""
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(50.0) / d_model))
		
		pos_emb = torch.zeros(1, max_len, d_model)
		
		pos_emb[0, :, 0::2] = torch.sin(position * div_term)   # Even indices: sin
		pos_emb[0, :, 1::2] = torch.cos(position * div_term)   # Odd indices: cos
		
		self.register_buffer('pos_emb', pos_emb)
	

	def forward(
		self,
		x: torch.Tensor
	) -> torch.Tensor:
		"""
		
		"""
		x = x + self.pos_emb
		x = self.dropout(x)
		
		return x




class RelativePositionEncoding(nn.Module):
	def __init__(self, d_model, max_len=100, k = None, dropout: float = 0.0, scaling_type: str = 'linear'):
		super().__init__()
		
		k = 1/np.sqrt(max_len) if k is None else k
		
		self.dropout = nn.Dropout(p=dropout)
		
		self.seq_len = max_len
		self.scaling_factor = k
		self.scaling_type = scaling_type
		
		# Create distance matrix: |i - j|
		i_indices = torch.arange(max_len).unsqueeze(1)
		j_indices = torch.arange(max_len).unsqueeze(0)
		distance = torch.abs(i_indices - j_indices).float()
		
		# Apply different scaling functions
		if scaling_type == 'linear':
			relative_emb = k * distance
		elif scaling_type == 'sqrt':
			relative_emb = k * torch.sqrt(distance + 1e-6)
		elif scaling_type == 'log':
			relative_emb = k * torch.log(distance + 1.0)
		elif scaling_type == 'inverse':
			relative_emb = k / (distance + 1.0)
		else:
			raise ValueError(f"Unknown scaling_type: {scaling_type}")
		
		# Register as buffer
		self.register_buffer('relative_emb', relative_emb)
	
	def forward(self, x):
		"""
		
		"""
		x = x + self.relative_emb
		x = self.dropout(x)
		
		return x