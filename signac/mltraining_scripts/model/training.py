"""
Training related functions
"""
from typing import Any
import json
import copy
import sys
import os

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

from mltraining_scripts.model.polymer_transformer_module import PolymerTransformerModule
from mltraining_scripts.model.losses import NLL, KLDivergence
from mltraining_scripts.post_processing.dataset_normalization import encode_batch
from mltraining_scripts.model.config_management import load_configs, setup_from_config


class EarlyStopping:
	"""
	Quick Early Stopping class, taken from previous project on uncertainty quantification:
	https://github.com/ctmht/task_adaptation_dl/
	"""
	
	def __init__(self, patience: int = 10) -> None:
		self.patience = patience
		self.last_improvement = 0
		self.best = 1e100

	def __call__(self, value: float) -> bool:
		""" returns True if the training should be stopped """
		# print("\n\n", value, "\n\n")
		if value < self.best:
			self.best = value
			self.last_improvement = 0
		else:
			self.last_improvement += 1

		return self.last_improvement >= self.patience

	def improves(self, new_value: float) -> bool:
		return new_value < self.best



def plot_stats(
	ptmodel: PolymerTransformerModule,
	cgns,
	first_attn_maps,
	first_input_str
):
	# Plot (clipped) gradient values
	plt.figure()
	
	x = np.arange(0, len(cgns), 1)
	plt.plot(x, cgns, color='purple', ls='--')
	# plt.plot(x, np.array(avggrads_noclip) * ptmodel.parameter_count(), color='blue')
	# plt.plot(x, np.array(avggrads_valclip) * ptmodel.parameter_count(), color='red')
	# plt.plot(x, np.array(avggrads_normclip) * ptmodel.parameter_count(), color='green')
	plt.show()
	
	# Plot attention maps for exemplary sequences
	cmap = 'RdPu_r'
	mha_layers = ptmodel.trunk_net.mha_layers
	num_heads = ptmodel.trunk_net.num_heads
	
	attn_maps = first_attn_maps
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
				labels = list(first_input_str.decode('utf-8')),
				size = 3
			)
			ax.set_yticks(
				ticks = np.arange(0, 100, 1) + 0.5,
				labels = list(first_input_str.decode('utf-8')),
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



def train_model(
	ptmodel: PolymerTransformerModule,
	input_data,
	output_data,
	val_input_data = None,
	val_output_data = None,
	**hyperparameters: dict[str, Any],
):
	"""
	
	"""
	# Generally used hyperparameters
	batch_size = hyperparameters.get('batch_size', 64)
	num_epochs = hyperparameters.get('num_epochs', 10)
	plot_stats = hyperparameters.get('plot_stats', False)
	
	l2reg_strength = hyperparameters.get('l2reg_strength', 0.0)
	
	# Training hyperparameters
	lr = hyperparameters.get('lr', 1e-4)
	clipgradnorm = hyperparameters.get('clipgradnorm', 4.0)
	# clipgradval = hyperparameters.get('clipgradval', torch.inf)
	
	early_stopping = hyperparameters.get('early_stopping', False)
	if early_stopping:
		if val_input_data is None or val_output_data is None:
			raise ValueError("Early stopping requires validation set")
		
		patience = hyperparameters.get('patience', 5)
		early_stopping_tracker = EarlyStopping(patience = patience)
	
	ptmodel.train()
	optimizer = torch.optim.Adam(ptmodel.parameters(), lr=lr, weight_decay=0.0)
	
	# Model device
	device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = torch.device(device_name)
	ptmodel = ptmodel.to(device)
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
	# losses_train = dict.fromkeys(['gte', 'pe', 'nbc', 'l2reg', 'total'], 0.0)
	# losses_val = dict.fromkeys(['gte', 'pe', 'nbc', 'l2reg', 'total'], 0.0)
	losskeys = ['gte', 'pe', 'nbc', 'l2reg', 'total']
	losses_train = dict([(losskey, []) for losskey in losskeys])
	losses_val = dict([(losskey, []) for losskey in losskeys])
	
	for epoch in range(num_epochs):
		np.random.shuffle(indices)
		
		epoch_loss = dict.fromkeys(losskeys, 0.0)
		
		cgns = []
		
		# Iterate through batches
		for batch_start in tqdm.tqdm(
			range(0, num_samples, batch_size),
			desc = f"Epoch {epoch+1 :>3}/{num_epochs :<3}"
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
			optimizer.zero_grad()
			
			pred = ptmodel(batch_input)
			
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
			loss_l2reg = ptmodel.parameters_l2norm()
			
			batch_loss = loss_gte + loss_pe + loss_nbc + l2reg_strength * loss_l2reg
			
			# Backward pass
			batch_loss.backward()
			
			# avggrads_bat = get_gradients()
			# avggrads_noclip.append(avggrads_bat.loc[0, 'avg_grad'].item())
			# torch.nn.utils.clip_grad_value_(ptmodel.parameters(), clipgradval)
			# avggrads_bat = get_gradients()
			# avggrads_valclip.append(avggrads_bat.loc[0, 'avg_grad'].item())
			# avggrads_bat = get_gradients()
			# avggrads_normclip.append(avggrads_bat.loc[0, 'avg_grad'].item())
			
			# Clip gradient by L^2 norm
			cgn = torch.nn.utils.clip_grad_norm_(ptmodel.parameters(), clipgradnorm)
			cgns.append(cgn)
			
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
		
		print(f"\tEpoch {epoch+1 :>3}/{num_epochs :<3} Losses: ", end='')
		print(*[key + ' ' + f"{avg_epoch_loss[key]:.6f}" for key in avg_epoch_loss], sep = ', ')
		
		if early_stopping:
			avg_val_loss = test_model(
				ptmodel,
				val_input_data,
				val_output_data,
				**hyperparameters
			)
			
			totalloss_val = avg_val_loss['total']
			
			if early_stopping_tracker.improves(totalloss_val):
				os.makedirs(
					os.path.join('data', 'tuning', hyperparameters['base_name']),
					exist_ok = True
				)
				ptmodel.save(
					os.path.join('data', 'tuning', hyperparameters['base_name'], hyperparameters['_specific_name'])
				)
			elif early_stopping_tracker(totalloss_val):
				break
		
		if plot_stats:
			plot_stats(
				ptmodel,
				cgns,
				pred['attn_maps'][0].detach().numpy(),
				batch_input_str[0]
			)
		
		for key in losses_train.keys():
			losses_train[key].append(avg_epoch_loss[key])
			losses_val[key].append(avg_val_loss[key])
	
	return losses_train, losses_val


def test_model(
	ptmodel: PolymerTransformerModule,
	input_data,
	output_data,
	**hyperparameters: dict[str, Any],
):
	"""
	
	"""
	# Generally used hyperparameters
	batch_size = hyperparameters.get('batch_size', 64)
	num_epochs = hyperparameters.get('num_epochs', 10)
	
	l2reg_strength = hyperparameters.get('l2reg_strength', 0.0)
	
	use_mcdropout = hyperparameters.get('use_mcdropout', False)
	num_epochs = 1 # Force one epoch
	
	if use_mcdropout:
		ptmodel.eval_mcdropout()
	else:
		ptmodel.eval()
	
	# Model device
	device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = torch.device(device_name)
	ptmodel = ptmodel.to(device)
	
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
		epoch_loss = dict.fromkeys(['gte', 'pe', 'nbc', 'l2reg', 'total'], 0.0)
		
		# Iterate through batches
		for batch_start in tqdm.tqdm(
			range(0, num_samples, batch_size),
			desc = f"Testing"
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
			pred = ptmodel(batch_input)
			
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
			loss_l2reg = ptmodel.parameters_l2norm()
			
			batch_loss = loss_gte + loss_pe + loss_nbc + l2reg_strength * loss_l2reg
			
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
		
		print(f"\tTested, Losses: ", end='')
		print(*[key + ' ' + f"{avg_epoch_loss[key]:.6f}" for key in avg_epoch_loss], sep = ', ')
		
		return avg_epoch_loss


def main_local():
	config_path = os.path.abspath(os.path.join('src', 'model', 'config.json'))
	configs = load_configs(path = config_path)
	
	passed_base_names = set()
	
	for config in configs:
		dataset, model, hyperparameters = setup_from_config(config)
		
		losses_train, losses_val = train_model(
			model,
			input_data = dataset['train_in'],
			output_data = dataset['train_out'],
			val_input_data = dataset['val_in'],
			val_output_data = dataset['val_out'],
			**hyperparameters
		)
		
		savepath = os.path.abspath(os.path.join('data', 'tuning', config['base_name'], config['base_name'])) + '.csv'
		
		result_ser = pd.Series(config)
		result_df = pd.DataFrame([result_ser.tolist()], columns = result_ser.index)
		
		for key_tr, val_tr in losses_train.items():
			result_df[key_tr + '_train'] = [val_tr]
		for key_val, val_val in losses_val.items():
			result_df[key_val + '_val'] = [val_val]
		
		if config['base_name'] not in passed_base_names:
			passed_base_names.add(config['base_name'])
			output_header = True
		else:
			output_header = False
		result_df.to_csv(savepath, mode = 'a', header = output_header)


def _saveloadconfig():
	config_path = os.path.abspath(os.path.join('src', 'model', 'config.json'))
	configs = load_configs(path = config_path)
	
	config = configs[0]
	
	config_w_results = copy.deepcopy(config)
	
	for tp in ['_train', '_val']:
		for stat in ['gte', 'pe', 'nbc', 'l2reg', 'total']:
			config_w_results[stat + tp] = list(np.random.rand(5))
	
	this_config_path = os.path.abspath(os.path.join('src', 'model', 'config_for_this_job.json'))
	
	json.dump(config_w_results, open(this_config_path, 'w'))
	
	loaded_config = json.load(open(this_config_path, 'r'))
	
	print(config_w_results, loaded_config, sep='\n\n')


def main(config_path: str, final_path: str):
	"""
	Main function modified for use on Habrok through parallel signac jobs
	"""
	config = json.load(open(config_path, 'r'))
	final = copy.deepcopy(config)
	dataset, model, hyperparameters = setup_from_config(config)
		
	losses_train, losses_val = train_model(
		model,
		input_data = dataset['train_in'],
		output_data = dataset['train_out'],
		val_input_data = dataset['val_in'],
		val_output_data = dataset['val_out'],
		**hyperparameters
	)
	
	for key_tr, val_tr in losses_train.items():
		final[key_tr + '_train'] = [val_tr]
	for key_val, val_val in losses_val.items():
		final[key_val + '_val'] = [val_val]
	
	json.dump(final, open(final_path, 'w'))


if __name__ == '__main__':
	job_config_json = sys.argv[1]
	job_final_json = sys.argv[2]
	
	main(job_config_json, job_final_json)