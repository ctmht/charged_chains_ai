from typing import Any

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch import optim
from torch import nn
import pandas as pd
import numpy as np
import torch
import tqdm

from matplotlib import pyplot as plt
import seaborn as sns

from model.polymer_transformer_module import PolymerTransformerModule
from model.losses import NLL, KLDivergence
from src.post_processing.dataset_normalization import encode_batch


class PolymerTransformer:
	""" Full model class containing training and inference """
	
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
		self.ptmodel = PolymerTransformerModule(
			embedding_dim = embedding_dim,
			num_heads = num_heads,
			dropout = dropout,
			mha_layers = mha_layers,
			hidden_dims_within = hidden_dims_within,
			hidden_dims_after = hidden_dims_after,
			temperature = temperature
		)
	
	
	def train_model(
		self,
		input_data,
		output_data,
		**hyperparameters: dict[str, Any]
	):
		"""
		
		"""
		return self._forwardpass_over_data(
			input_data,
			output_data,
			training = True,
			**hyperparameters
		)
	
	
	def test_model(
		self,
		input_data,
		output_data,
        **hyperparameters: dict[str, Any]
	):
		"""
		
		"""
		return self._forwardpass_over_data(
			input_data,
			output_data,
			training = False,
			**hyperparameters
		)
	
	
	
	def _forwardpass_over_data(
		self,
		input_data,
		output_data,
		training: bool = False,
		**hyperparameters: dict[str, Any],
	):
		"""
		
		"""
		# Generally used hyperparameters
		batch_size = hyperparameters.get('batch_size', 64)
		num_epochs = hyperparameters.get('num_epochs', 10)
		plot_stats = hyperparameters.get('plot_stats', False)
		
		l2reg_strength = hyperparameters.get('l2reg_strength', 0.0)
		
		if not training:
			# Test hyperparameters
			use_mcdropout = hyperparameters.get('use_mcdropout', False)
			num_epochs = 1 # Force one epoch
			
			if use_mcdropout:
				self.ptmodel.eval_mcdropout()
			else:
				self.ptmodel.eval()
		else:
			# Training hyperparameters
			lr = hyperparameters.get('lr', 1e-4)
			clipgradval = hyperparameters.get('clipgradval', torch.inf)
			clipgradnorm = hyperparameters.get('clipgradnorm', 4.0)
			
			early_stopping = hyperparameters.get('early_stopping', False)
			if early_stopping:
				val_input_data = hyperparameters['val_input_data']
				val_output_data = hyperparameters['val_output_data']
			
			self.ptmodel.train()
			optimizer = torch.optim.Adam(self.ptmodel.parameters(), lr=lr, weight_decay=0.0)
		
		# Model device
		device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
		device = torch.device(device_name)
		self.ptmodel = self.ptmodel.to(device)
		print(device_name)
		
        # Extract targets, convert to PyTorch tensors
		target = output_data
		
		for key in target:
			if isinstance(target[key], np.ndarray):
				target[key] = torch.from_numpy(target[key]).float()
		
		num_samples = len(input_data)
		indices = np.arange(num_samples)
		
		# Losses definitions
		kldivlossmodule = KLDivergence()
		nlllossmodule = NLL()
		
		# Iterate through epochs
		for epoch in range(num_epochs):
			if training:
				np.random.shuffle(indices)
			
			epoch_loss = dict.fromkeys(['gte', 'pe', 'nbc', 'l2reg', 'total'], 0.0)
			
			avggrads_noclip = []
			avggrads_normclip = []
			avggrads_valclip = []
			cgns = []
			
			# Iterate through batches
			for batch_start in tqdm.tqdm(
				range(0, num_samples, batch_size),
				desc = f"Epoch {epoch+1}" if training else "Testing"
			):
				# Prepare batch
				batch_end = min(batch_start + batch_size, num_samples)
				batch_indices = indices[batch_start:batch_end]
				
				batch_input_str = input_data[batch_indices]
				batch_input = encode_batch(batch_input_str).to(device)
				
				batch_target = {}
				for key in target:
					batch_target[key] = target[key][batch_indices].to(device)
				
				# Forward pass
				if training:
					optimizer.zero_grad()
				pred = self.ptmodel(batch_input)
				
				loss_gte = kldivlossmodule(
					pred_mean = pred['gte_oem'],
					pred_cov = pred['gte_oev'],
					true_mean = batch_target['gte_oem'],
					true_cov = batch_target['gte_oev'],
					batch_reduction = 'mean'
				)
				loss_pe = kldivlossmodule(
					pred_mean = pred['pe_mean'],
					pred_cov = torch.square(pred['pe_std']).unsqueeze(-1),
					true_mean = batch_target['pe_mean'].unsqueeze(-1),
					true_cov = torch.square(batch_target['pe_std']).unsqueeze(-1).unsqueeze(-1),
					batch_reduction = 'mean'
				)
				loss_nbc = nlllossmodule(
					mean = pred['nbc_mean'],
					cov = torch.diag_embed(pred['nbc_var']),
					target = batch_target['nbc'],
					batch_reduction = 'mean'
				)
				
				# L2 regularization loss
				loss_l2reg = 0
				for name, param in self.ptmodel.named_parameters():
					if 'linear' in name:
						loss_l2reg = loss_l2reg + param.norm(2)
				
				batch_loss = 10 * loss_gte + 10 * loss_pe + 10 * loss_nbc + l2reg_strength * loss_l2reg
				
				# Backward pass
				if training:
					batch_loss.backward()
					
					avggrads_bat = self.get_gradients()
					avggrads_noclip.append(avggrads_bat.loc[0, 'avg_grad'].item())
					
					torch.nn.utils.clip_grad_value_(self.ptmodel.parameters(), clipgradval)
					
					avggrads_bat = self.get_gradients()
					avggrads_valclip.append(avggrads_bat.loc[0, 'avg_grad'].item())
					
					cgn = torch.nn.utils.clip_grad_norm_(self.ptmodel.parameters(), clipgradnorm)
					cgns.append(cgn)
					
					avggrads_bat = self.get_gradients()
					avggrads_normclip.append(avggrads_bat.loc[0, 'avg_grad'].item())
					
					optimizer.step()
				
				# Save losses of this batch
				epoch_loss['gte'] += loss_gte.item()
				epoch_loss['pe'] += loss_pe.item()
				epoch_loss['nbc'] += loss_nbc.item()
				epoch_loss['l2reg'] += l2reg_strength * loss_l2reg.item()
				epoch_loss['total'] += batch_loss.item()
			
			# Save losses of this epoch
			avg_epoch_loss = {}
			for key in epoch_loss:
				avg_epoch_loss[key] = epoch_loss[key] / num_samples
			
			print()
			print(f"Epoch {epoch+1}/{num_epochs}, Clipgradnorm: {np.sum(cgns)/num_samples:.6f}, Losses: ", end='')
			print(*[key + ' ' + f"{avg_epoch_loss[key]:.6f}" for key in avg_epoch_loss], sep = ', ')
			print()
			
			if training:
				if early_stopping:
					self.test_model(
						val_input_data,
						val_output_data,
						**hyperparameters
					)
			
			if plot_stats:
				# Plot (clipped) gradient values
				plt.figure()
				
				x = np.arange(0, len(cgns), 1)
				plt.plot(x, cgns, color='purple', ls='--')
				plt.plot(x, np.array(avggrads_noclip) * self.count_parameters(), color='blue')
				plt.plot(x, np.array(avggrads_valclip) * self.count_parameters(), color='red')
				plt.plot(x, np.array(avggrads_normclip) * self.count_parameters(), color='green')
				plt.show()
				
				# Plot attention maps for exemplary sequences
				cmap = 'RdPu_r'
				mha_layers = self.ptmodel.trunk_net.mha_layers
				num_heads = self.ptmodel.trunk_net.num_heads
				
				attn_maps = pred['attn_maps'][0].detach().numpy()
				global_min = attn_maps.min()
				global_max = attn_maps.max()
				
				for layer in range(mha_layers):
					fig, axs = plt.subplots(
						1,
						num_heads,
						figsize = (num_heads * 4 + 0.5, 4),
						gridspec_kw = {'wspace': 0},
						dpi = 400
					)
					axs = axs.flatten()
					
					for head in range(num_heads):
						ax = axs[head]
						
						attn_map = attn_maps[layer][head]
						sns.heatmap(
							attn_map,
							ax = ax,
							cbar = False,
							cmap = cmap,
							vmin = global_min,
							vmax = global_max
						)
						
						ax.set_xticks(
							ticks = np.arange(0, 100, 1) + 0.5,
							labels = list(batch_input_str[0].decode('utf-8')),
							size = 3
						)
						ax.set_yticks(
							ticks = np.arange(0, 100, 1) + 0.5,
							labels = list(batch_input_str[0].decode('utf-8')),
							size = 3
						)
						ax.set_aspect(aspect = 'equal')
						ax.xaxis.set_ticks_position('top')
					
					cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
					sm = plt.cm.ScalarMappable(
						cmap = cmap,
						norm = plt.Normalize(vmin = global_min, vmax = global_max)
					)
					sm.set_array([])
					plt.colorbar(sm, cax = cbar_ax)
					
					plt.show()
		
	
	def get_gradients(
		self
	):
		"""
		
		"""
		avg_grad_df = []
		
		for idx, obj in enumerate(self.ptmodel.named_modules()):
			if obj[1].parameters() is not None:
				grads = [par.grad for par in obj[1].parameters() if par.grad is not None]
				grads_norm = nn.utils.get_total_norm(grads)
				
				if grads_norm != 0:
					grads_size = sum([
						pgrad.numel() for pgrad in grads if pgrad.abs().sum().item() != 0
					])
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



if __name__ == '__main__':
	from src.post_processing.dataset_normalization import create_datasets
	
	import os
	DATA_PATH = os.path.abspath("././data/full_fixed_results.h5")
	print(DATA_PATH)
	
	dataset_splits = create_datasets(
		DATA_PATH,
		zscore_norm=False,
		minmax_norm=True,
		test_prop=0.2,
		val_prop=0.2,
		newmin = 1e-6,
		newmax = 1
	)
	
	EMBEDDING_DIM = 32
	NUM_HEADS = 4
	MHA_LAYERS = 2

	model = PolymerTransformer(
		embedding_dim = EMBEDDING_DIM,
		num_heads = NUM_HEADS,
		mha_layers = MHA_LAYERS,
		dropout = 0.2,
		hidden_dims_within = [64, 64],
		hidden_dims_after = [64],
		temperature = 1
	)
	print(f"\nThe model has {model.count_parameters()} parameters")
	
	model.train_model(
		input_data = dataset_splits['train_in'],
		output_data = dataset_splits['train_out'],
		lr = 5e-4,
		batch_size = 256,
		num_epochs = 20,
		l2reg_strength = 0.05,
		early_stopping = True,
		val_input_data = dataset_splits['val_in'],
		val_output_data = dataset_splits['val_out'],
	)