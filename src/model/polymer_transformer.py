from typing import Any

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch import optim
from torch import nn
import pandas as pd
import numpy as np
import torch
import h5py
import tqdm

from matplotlib import pyplot as plt
import seaborn as sns

from model.polymer_transformer_module import PolymerTransformerModule


class PolymerTransformer:
	""" Full model class containing training and inference """
	
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
		self.ptmodel = PolymerTransformerModule(
			embedding_dim = embedding_dim,
			num_heads = num_heads,
			dropout = dropout,
			mha_layers = mha_layers,
			hidden_dims = hidden_dims
		)

	
	
	def train_model(
		self,
		input_data,
		output_data,
        **hyperparameters: dict[str, Any]
	):
		"""
		
		"""
        # Extract target means and covariances
		target = output_data
		
		# Convert to PyTorch tensors
		for key in target:
			target[key] = torch.from_numpy(target[key]).float()
		
		# Training hyperparameters
		lr = hyperparameters['lr']
		l2reg_strength = hyperparameters['l2reg_strength']
		num_epochs = hyperparameters['num_epochs']
		batch_size = hyperparameters['batch_size']
		
		# Training parameters
		num_samples = len(input_data)
		indices = np.arange(num_samples)

		# Setup training
		device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
		device = torch.device(device_name)
		print(device_name)
		
		self.ptmodel = self.ptmodel.to(device)
		optimizer = torch.optim.Adam(self.ptmodel.parameters(), lr=lr, weight_decay=0.0)
		
		for epoch in range(num_epochs):
			np.random.shuffle(indices)
			
			epoch_loss = dict.fromkeys(['gte', 'pe', 'nbc', 'l2reg', 'total'], 0.0)
			
			avggrads = []
			
			for batch_start in tqdm.tqdm(range(0, num_samples, batch_size), desc = f"Epoch {epoch+1}"):
				batch_end = min(batch_start + batch_size, num_samples)
				batch_indices = indices[batch_start:batch_end]
				
				# Prepare batch
				batch_input_str = input_data[batch_indices]
				batch_input = self.encode_batch(batch_input_str).to(device)
				
				batch_target = {}
				for key in target:
					batch_target[key] = target[key][batch_indices].to(device)
				
				# Forward pass
				optimizer.zero_grad()
				pred = self.ptmodel(batch_input)
				
				# Compute losses: KL Divergence for gte and pe, GNLL for nbc
				for batit in range(batch_end - batch_start):
					# Gyration tensor eigenvalues: KL Divergence
					pred_dist = MultivariateNormal(
						loc = pred['gte_oem'][batit],
						covariance_matrix = pred['gte_oev'][batit]
					)
					true_dist = MultivariateNormal(
						loc = batch_target['gte_oem'][batit],
						covariance_matrix = batch_target['gte_oev'][batit]
					)
					loss_gte_batit = torch.distributions.kl.kl_divergence(true_dist, pred_dist)
					loss_gte = loss_gte_batit + (loss_gte if batit > 0 else 0)

					# Potential energy: KL Divergence
					pred_dist = Normal(
						loc = pred['pe_mean'][batit],
						scale = pred['pe_var'][batit]
					)
					true_dist = Normal(
						loc = batch_target['pe_mean'][batit],
						scale = batch_target['pe_var'][batit]
					)
					loss_pe_batit = torch.distributions.kl.kl_divergence(true_dist, pred_dist)
					loss_pe = loss_pe_batit + (loss_pe if batit > 0 else 0)
					
					loss_nbc_batit = torch.nn.GaussianNLLLoss(full = True)(
						pred['nbc_mean'][batit],
						batch_target['nbc'][batit],
						pred['nbc_var'][batit]
					)
					loss_nbc = loss_nbc_batit + (loss_nbc if batit > 0 else 0)
				
				# L2 regularization loss
				loss_l2reg = 0
				for name, param in self.ptmodel.named_parameters():
					if 'linear' in name:
						loss_l2reg = loss_l2reg + param.norm(2)    
				
				loss_gte /= batch_size
				loss_pe /= batch_size
				loss_nbc /= batch_size
				batch_loss = loss_gte + loss_pe + loss_nbc + l2reg_strength * loss_l2reg
				
				# Backward pass
				batch_loss.backward()
				
				avggrads_bat = self.get_gradients()
				avggrads.append(avggrads_bat.loc[0, 'avg_grad'].item())
				
				torch.nn.utils.clip_grad_value_(self.ptmodel.parameters(), 10.0)
				cgn = torch.nn.utils.clip_grad_norm_(self.ptmodel.parameters(), 1.0)
						
				optimizer.step()
				
				epoch_loss['gte'] += loss_gte.item()
				epoch_loss['pe'] += loss_pe.item()
				epoch_loss['nbc'] += loss_nbc.item()
				epoch_loss['l2reg'] += l2reg_strength * loss_l2reg.item()
				epoch_loss['total'] += batch_loss.item()


			avg_epoch_loss = {}
			for key in epoch_loss:
				avg_epoch_loss[key] = epoch_loss[key] / num_samples
			
			print(f"Epoch {epoch+1}/{num_epochs}, Clipgradnorm: {cgn:.6f}, Losses: ", end='')
			print(*[key + ' ' + f"{avg_epoch_loss[key]:.6f}" for key in avg_epoch_loss], sep = ', ', end = '\b\b ')
			
			plt.figure()
			plt.plot(avggrads)
			plt.show()
			
			mha_layers = self.ptmodel.trunk_net.mha_layers
			num_heads = self.ptmodel.trunk_net.num_heads
			for layer in range(mha_layers):
				_, axs = plt.subplots(1, num_heads, figsize = (num_heads * 5, 4), gridspec_kw={'wspace': 0}, dpi=400)
				axs = axs.flatten()
				
				# spectra = []
				for head in range(num_heads):
					ax = axs[head]
					
					attn_map = pred['attn_maps'][0][layer][head].detach().numpy()
					# spectrum = np.linalg.svd(attn_map, full_matrices = False, compute_uv = False, hermitian = False)
					# spectra.append(spectrum[spectrum != 0])
					
					sns.heatmap(attn_map, ax=ax)
					
					ax.set_xticks(ticks=np.arange(0, 100, 1) + 0.5, labels=list(batch_input_str[0].decode('utf-8')), size=3)
					ax.set_yticks(ticks=np.arange(0, 100, 1) + 0.5, labels=list(batch_input_str[0].decode('utf-8')), size=3)
					ax.set_aspect(aspect = 'equal')
					ax.xaxis.set_ticks_position('top')
				# print(*spectra, sep='\n')
				plt.show()
	
	
	def get_gradients(
		self
	):
		"""
		
		"""
		avg_grad_df = []
		
		for idx, obj in enumerate(self.ptmodel.named_modules()):
			if obj[1].parameters() is not None:
				grads = [par.grad for par in obj[1].parameters()]
				grads_norm = nn.utils.get_total_norm(grads)
		
				if grads_norm != 0:
					grads_size = sum([par.grad.numel() for par in obj[1].parameters() if par.grad.abs().sum().item() != 0])
					avg_grad = nn.utils.get_total_norm(grads) / grads_size
					avg_grad_df.append([
						obj[0],
						avg_grad.item()
					])
		
		avg_grad_df = pd.DataFrame(avg_grad_df, columns = ['name', 'avg_grad'])
		
		return avg_grad_df
	
	
	def count_parameters(
		self
	) -> int:
		"""
		
		"""
		return sum(p.numel() for p in self.ptmodel.parameters() if p.requires_grad)


	def encode_batch(
		self,
		batch_strings: np.ndarray
	) -> torch.Tensor:
		"""
		Vectorized one-hot encoding of batch of byte strings
		"""
		# Convert byte strings to Python strings
		char_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
		str_list = [s.decode('utf-8') for s in batch_strings]
		
		# Create integer indices using numpy vectorized operations
		# Stack all strings into a single array for vectorized processing
		batch_size = len(str_list)
		indices = np.zeros((batch_size, 100), dtype=np.int64)
		
		for i, s in enumerate(str_list):
			for j, char in enumerate(s):
				indices[i, j] = char_to_idx[char]
		
		# Convert to one-hot encoding using torch
		indices_tensor = torch.from_numpy(indices).long()
		one_hot = torch.nn.functional.one_hot(indices_tensor, num_classes=4).float()
		return one_hot



if __name__ == '__main__':
	import os
	DATA_PATH = os.path.abspath("././data/full_results_n2.h5")
	
	model = PolymerTransformer(
		hidden_dims = [], dropout = 0.2, mha_layers = 2
	)
	print(model.count_parameters())
	
	with h5py.File(DATA_PATH, "r") as fin:
		train_size = int(0.8 * fin['indexers/sequence'][:].shape[0])
		
		model.train_model(
			input_data = fin['indexers/sequence'][:train_size],
			output_data = {
				'gte_oem': fin['shape_descriptors/gyr_tensor_oevals/gyr_tensor_oevals_perfm'][:train_size],
				'gte_oev': fin['shape_descriptors/gyr_tensor_oevals/gyr_tensor_oevals_perfv'][:train_size],
				'pe_mean': fin['shape_descriptors/poteng/poteng_perfm'][:train_size],
				'pe_var': fin['shape_descriptors/poteng/poteng_perfs'][:train_size],
				'nbc': np.vstack([
					fin['shape_descriptors/nbc_perbm/nbc_perbm_A'][:train_size],
					fin['shape_descriptors/nbc_perbm/nbc_perbm_B'][:train_size],
					fin['shape_descriptors/nbc_perbm/nbc_perbm_C'][:train_size],
					fin['shape_descriptors/nbc_perbm/nbc_perbm_D'][:train_size]
				]).T
			},
			lr = 1e-3,
			batch_size = 32,
			num_epochs = 5,
			l2_reg = 0.125
		)