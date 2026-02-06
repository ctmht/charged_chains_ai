from torch import nn
import numpy as np
import spdlayers
import torch

from mltraining_scripts.model.trunk_net_module import TrunkNetModule


class PolymerTransformerModule(nn.Module):
	""" Transformer-based sequence processing model (nn.Module subclass) """
	
	def __init__(
		self,
		embedding_dim: int = 32,
		num_heads: int = 4,
		dropout: float = 0.0,
		mha_layers: int = 2,
		hidden_dims_within: list[int] = None,
		hidden_dims_after: list[int] = None,
		temperature: float = 1.0
	):
		"""
		
		"""
		super(PolymerTransformerModule, self).__init__()
		
		self._mcdropout = False

		# hidden_dims.insert(0, embedding_dim)
		hidden_dims_after.insert(0, embedding_dim)
		
		self.trunk_net = TrunkNetModule(
			embedding_dim = embedding_dim,
			num_heads = num_heads,
			dropout = dropout,
			mha_layers = mha_layers,
			hidden_dims_within = hidden_dims_within,
			hidden_dims_after = hidden_dims_after,
			temperature = temperature
		)

		# Gyration tensor eigenvalues
		gte_head_dim = 16
		self.gte_head = nn.Sequential(
			nn.Linear(hidden_dims_after[-1], gte_head_dim),   # transform pooled embeddings
			nn.LayerNorm(gte_head_dim),                       # add (gradient) stability
			nn.ReLU()
		)
		self.gte_oem_head = nn.Sequential(
			nn.Linear(gte_head_dim, 3),
			nn.Sigmoid()
		)
		self.gte_oev_head = nn.Sequential(
			# output the 6 independent entries of covar + enforce symmetric positive definiteness
			nn.Linear(gte_head_dim, spdlayers.in_shape_from(3)),
			spdlayers.Cholesky(3, positive = 'Softplus')
		)

		# Potential energy
		poteng_head_dim = 8
		self.poteng_head = nn.Sequential(
			nn.Linear(hidden_dims_after[-1], poteng_head_dim),# transform pooled embeddings
			nn.LayerNorm(poteng_head_dim),                    # add (gradient) stability
			nn.ReLU()
		)
		self.poteng_mean_head = nn.Sequential(
			nn.Linear(poteng_head_dim, 1),
			nn.Sigmoid()
		)
		self.poteng_std_head = nn.Sequential(
			nn.Linear(poteng_head_dim, 1),
			nn.Sigmoid()
		)

		# Mean neighbour counts
		nbc_head_dim = 16
		self.nbc_head = nn.Sequential(
			nn.Linear(hidden_dims_after[-1], nbc_head_dim),   # transform pooled embeddings
			nn.LayerNorm(nbc_head_dim),                       # add (gradient) stability
			nn.ReLU(),
		)
		self.nbc_mean_head = nn.Sequential(
			nn.Linear(nbc_head_dim, 4),
			nn.Sigmoid()
		)
		self.nbc_var_head = nn.Sequential(
			nn.Linear(nbc_head_dim, 4),
			nn.Softplus()
		)
		
		self._init_weights()
	
	
	def _init_weights(
		self,
	):
		for name, module in self.named_modules():
			if isinstance(module, spdlayers.Cholesky):
				continue
			elif isinstance(module, nn.LayerNorm):
				nn.init.ones_(module.weight)
				nn.init.zeros_(module.bias)
			elif isinstance(module, nn.MultiheadAttention) or isinstance(module, nn.Linear):
				if hasattr(module, 'bias') and module.bias is not None:
					nn.init.zeros_(module.bias)
				for paramname, param in module.named_parameters():
					if 'q_proj' in paramname or 'k_proj' in paramname:
						d_k = self.trunk_net.embedding_dim // self.trunk_net.num_heads
						std = 1.0 / np.sqrt(d_k)
						nn.init.normal_(param, mean=0.0, std=std)
					elif 'in_proj' in paramname:
						d_k = self.trunk_net.embedding_dim // self.trunk_net.num_heads
						std = 1.0 / np.sqrt(d_k)
						nn.init.normal_(param, mean=0.0, std=std)
					elif 'v_proj' in paramname or 'out_proj' in paramname:
						nn.init.xavier_uniform_(param)
					else:
						nn.init.normal_(param, mean=0.0, std=0.2)

		

	def forward(
		self,
		data
	):
		pooled_emb, attn_maps = self.trunk_net(data)

		# Gyration tensor ordered eigenvalues mean + covariance heads
		_gte_out = self.gte_head(pooled_emb)
		_gte_oem_out = self.gte_oem_head(_gte_out)
		_gte_oev_out = self.gte_oev_head(_gte_out)

		# Potential energy mean + std dev head
		_poteng_head = self.poteng_head(pooled_emb)
		_poteng_mean_out = self.poteng_mean_head(_poteng_head)
		_poteng_std_out = self.poteng_std_head(_poteng_head)
		
		# Mean neighbour counts head
		_nbc_out = self.nbc_head(pooled_emb)
		_nbc_mean_out = self.nbc_mean_head(_nbc_out)
		_nbc_var_out = torch.square(self.nbc_var_head(_nbc_out))

		return {
			'gte_oem': _gte_oem_out,
			'gte_oev': _gte_oev_out,
			'pe_mean': _poteng_mean_out,
			'pe_std': _poteng_std_out,
			'nbc_mean': _nbc_mean_out,
			'nbc_var': _nbc_var_out,
			'pooled_emb': pooled_emb,
			'attn_maps': attn_maps
		}
	
	
	def train(
		self,
		mode: bool = True
	) -> None:
		"""
		
		"""
		super().train(mode = mode)
		self._mcdropout = False
	
	
	def eval_mcdropout(
		self
	) -> None:
		"""
		
		"""
		super().eval()
		self._mcdropout = True
	
	
	def count_parameters(
		self
	) -> int:
		"""
		
		"""
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
	
	
	def parameters_l2norm(
		self
	) -> float:
		"""
		
		"""
		norm = 0
		for name, param in self.named_parameters():
			if 'linear' in name:
				norm = norm + param.norm(2)
		return norm
	
	
	def save(
		self,
		path: str
	):
		"""
		
		"""
		torch.save(self, path)
	
	
	@classmethod
	def load(
		cls,
		path: str
	):
		"""
		
		"""
		return torch.load(path, weights_only = False)