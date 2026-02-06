"""
Module for managing JSON model configuration files. Taken from previous project on uncertainty
quantification at https://github.com/ctmht/task_adaptation_dl/
"""
from copy import deepcopy
import json
import os

from src.model.polymer_transformer_module import PolymerTransformerModule
from src.post_processing.dataset_normalization import create_datasets


DEFAULT_CONFIG = {
	'zscore_norm': False,
	'minmax_norm': True,
	'val_prop': 0.2,
	'test_prop': 0.2,
	'newmin': 1e-6,
	'newmax': 1,
	
	'embedding_dim': 64,
	'mha_layers': 2,
	'num_heads': 4,
	'dropout': 0.3,
	'hidden_dims_within': [64, 64],
	'hidden_dims_after': [64],
	
	'batch_size': 64,
	'num_epochs': 50,
	'l2reg_strength': 0.02,
	'lr': 1e-3,
	'clipgradnorm': 4.0,
	
	'early_stopping': True,
	'patience': 5,
	
	'use_mcdropout': False,
	
	'plot_stats': False,
}


def load_configs(path: str) -> list[dict]:
	meta_config = json.load(open(path, "r"))
	configs = []
	for i in meta_config["experiments"]:
		if "base_name" not in i:
			raise ValueError("Each individual config must contain a 'base_name'")
		configs += vary_lists_configs(i, meta_config["do_not_vary"])
	
	print(f"Created {len(configs)} experimental configurations")
	return configs


def vary_lists_configs(
	config, leave_out: list[str], variations: dict | None = None
) -> list[dict]:
	configs = []
	variations = variations or {}
	for k, v in config.items():
		if isinstance(v, list) and k not in leave_out:
			new_leave_out = deepcopy(leave_out)
			new_leave_out.append(k)
			for i in v:
				# print(k, i)
				config_copy = deepcopy(config)
				config_copy[k] = i
				variations[str(k)] = str(i)
				configs += vary_lists_configs(config_copy, new_leave_out, variations)
			return configs

	if not variations:
		specific_name = config["base_name"]
	else:
		specific_name = "-".join(f"{k}={v}" for k, v in variations.items())
		specific_name = specific_name.replace(" ", "")  # just in case
	config["_specific_name"] = specific_name
	return [set_defaults(config)]


def set_defaults(config: dict):
	default_config = deepcopy(DEFAULT_CONFIG)
	default_config.update(config)
	return default_config


def setup_from_config(
	config: dict
) -> tuple[dict, PolymerTransformerModule, dict]:
	"""
	
	"""
	print("Running experiment from config:", config, sep='\n')
	
	dataset = create_datasets(
		os.path.join('data', 'full_fixed_results.h5'),
		zscore_norm = config['zscore_norm'],
		minmax_norm = config['minmax_norm'],
		test_prop = config['test_prop'],
		val_prop = config['val_prop'],
		newmin = config['newmin'],
		newmax = config['newmax']
	)
	
	model = PolymerTransformerModule(
		embedding_dim = config['embedding_dim'],
		num_heads = config['num_heads'],
		dropout = config['dropout'],
		mha_layers = config['mha_layers'],
		hidden_dims_within = config['hidden_dims_within'],
		hidden_dims_after = config['hidden_dims_after'],
	)
	model.train()
	
	print("Model parameters count:", model.count_parameters())
	print("Dataset train set #items:", dataset['train_in'].shape[0])
	
	hyperparameters = dict(**config)
	
	return dataset, model, hyperparameters