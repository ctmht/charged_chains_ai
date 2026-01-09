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
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(100.0) / d_model))
		
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