from typing import Optional

import numpy as np
import torch
import h5py


# TODO: reconstruction


INPUT_DSNAMES = [
	'indexers/sequence'
]
OUTPUT_DSNAMES = [
	'shape_descriptors/gyr_tensor_oevals/gyr_tensor_oevals_perfm',
	'shape_descriptors/gyr_tensor_oevals/gyr_tensor_oevals_perfv',
	'shape_descriptors/poteng/poteng_perfm',
	'shape_descriptors/poteng/poteng_perfs',
	'shape_descriptors/nbc/nbc_lastf',
]


def minmax_normalize(
	series: np.ndarray,
	newmin: float = 0.0,
	newmax: float = 1.0,
	oldmin: Optional[np.ndarray] = None,
	oldmax: Optional[np.ndarray] = None,
	**kwargs,
) -> tuple[np.ndarray, float, float]:
	if oldmin is None and oldmax is None:
		oldmin = series.min(axis = 0)
		oldmax = series.max(axis = 0)
	minmax_norm_series = (series - oldmin) / (oldmax - oldmin) * newmax + newmin
	
	return minmax_norm_series, oldmin, oldmax


def zscore_normalize(
	series: np.ndarray,
	oldmean: Optional[np.ndarray] = None,
	oldstd: Optional[np.ndarray] = None,
	**kwargs,
) -> tuple[np.ndarray, float, float]:
	if oldmean is None and oldstd is None:
		oldmean = series.mean(axis = 0)
		oldstd = series.std(axis = 0, ddof = 1)
	zscore_norm_series = (series - oldmean) / oldstd

	return zscore_norm_series, oldmean, oldstd


def normalize_outputs(
	output_dsets: dict[str, np.ndarray],
	zscore_norm: bool = True,
	minmax_norm: bool = True,
	recons: Optional[dict] = None,
	**kwargs,
) -> tuple[dict[str, np.ndarray] | dict[str, np.ndarray | None]]:
	"""
	Applies normalization to each entry

	If both zscore and minmax are True, then zscore is always applied first
	"""
	outputs_zsn = {}
	
	if zscore_norm:
		oldmean = {}
		oldstd = {}
		for key in output_dsets:
			if 'gte_oev' in key:
				outputs_zsn[key] = output_dsets[key]
				oldmean[key] = None
				oldstd[key] = None
			else:
				outputs_zsn[key], oldmean[key], oldstd[key] = zscore_normalize(
					output_dsets[key],
					oldmean = recons['oldmean'][key] if recons else None,
					oldstd = recons['oldstd'][key] if recons else None,
				)
	else:
		for key in output_dsets:
			outputs_zsn[key] = output_dsets[key]
	
	outputs_mmn = {}
	if minmax_norm:
		oldmin = {}
		oldmax = {}
		# oldtriudxm = {}
		for key in output_dsets:
			if 'gte_oev' in key:
				outputs_mmn[key] = outputs_zsn[key]
				oldmin[key] = None
				oldmax[key] = None
			else:
				outputs_mmn[key], oldmin[key], oldmax[key] = minmax_normalize(
					outputs_zsn[key],
					oldmin = recons['oldmin'][key] if recons else None,
					oldmax = recons['oldmax'][key] if recons else None,
					newmin = kwargs['newmin'] if 'newmin' in kwargs else None,
					newmax = kwargs['newmax'] if 'newmax' in kwargs else None,
				)
	else:
		for key in output_dsets:
			outputs_mmn[key] = outputs_zsn[key]

	reconstructors = {
		'oldmin': oldmin if minmax_norm else None,
		'oldmax': oldmax if minmax_norm else None,
		'oldmean': oldmean if zscore_norm else None,
		'oldstd': oldstd if zscore_norm else None,
	}
	
	return outputs_mmn, reconstructors


def train_val_test_split(
	input_dsets: dict[str, np.ndarray],
	output_dsets: dict[str, np.ndarray],
	test_prop: float = 0.2,
	val_prop: float = 0.0
) -> tuple[dict[str, np.ndarray]]:
	"""
	Splits dataset into train, (optionally) validation, and test set
	"""
	total = output_dsets[list(output_dsets.keys())[0]].shape[0]
	test_size = int(test_prop * total)
	train_size = total - test_size
	
	train_input_dsets = {}
	train_output_dsets = {}
	test_input_dsets = {}
	test_output_dsets = {}
	
	if val_prop > 0:
		val_size = int(val_prop * total)
		train_size = total - test_size - val_size
		
		val_input_dsets = {}
		val_output_dsets = {}
	
	for key in input_dsets:
		train_input_dsets[key] = input_dsets[key][ : train_size]
		
		if val_prop > 0:
			val_input_dsets[key] = input_dsets[key][train_size : (train_size + val_size)]
			test_input_dsets[key] = input_dsets[key][(train_size + val_size) : ]
		else:
			test_input_dsets[key] = input_dsets[key][train_size : ]
	
	for key in output_dsets:
		train_output_dsets[key] = output_dsets[key][ : train_size]

		if val_prop > 0:
			val_output_dsets[key] = output_dsets[key][train_size : (train_size + val_size)]
			test_output_dsets[key] = output_dsets[key][(train_size + val_size) : ]
		else:
			test_output_dsets[key] = output_dsets[key][train_size : ]

	if val_prop > 0:
		return {
			'inputs': (train_input_dsets, val_input_dsets, test_input_dsets),
			'outputs': (train_output_dsets, val_output_dsets, test_output_dsets)
		}
	return {
		'inputs': (train_input_dsets, None, test_input_dsets),
		'outputs': (train_output_dsets, None, test_output_dsets)
	}


def build_renamed(
	output_dsets: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
	"""
	Does two things: renames the keys as they are used by the model *and* stacks the four per-bead
	mean neighbour counts into a single (dataset_size, 4)-shaped output
	"""
	
	return {
		'gte_oem': output_dsets['shape_descriptors/gyr_tensor_oevals/gyr_tensor_oevals_perfm'],
		'gte_oev': output_dsets['shape_descriptors/gyr_tensor_oevals/gyr_tensor_oevals_perfv'],
		'pe_mean': output_dsets['shape_descriptors/poteng/poteng_perfm'],
		'pe_std': output_dsets['shape_descriptors/poteng/poteng_perfs'],
		'nbc': np.mean(output_dsets['shape_descriptors/nbc/nbc_lastf'], axis = 1)
	}


def encode_batch(
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


def create_datasets(
	datapath: str,
	zscore_norm: bool = False,
	minmax_norm: bool = True,
	test_prop: float = 0.2,
	val_prop: float = 0.2,
	**kwargs,
) -> dict[str, dict[str, np.ndarray] | dict[str, dict[str, np.ndarray | None]]]:
	"""
	Return:
		inputs: dict[str, np.ndarray]
		train_set: dict[str, np.ndarray]
		val_set: dict[str, np.ndarray]
		test_set: dict[str, np.ndarray]
		reconstruct: dict[str, dict[str, np.ndarray | None]]
			oldmins: oldmins if minmax_norm else None
			oldmaxs: oldmaxs if minmax_norm else None
			oldmeans: oldmeans if zscore_norm else None
			oldstds: oldstds if zscore_norm else None
			^ each of these is thus dict[str, np.ndarray | None]
	"""
	data = h5py.File(datapath, "r")
	
	input_ds = dict(zip(
		INPUT_DSNAMES,
		[data[dsname][:] for dsname in INPUT_DSNAMES]
	))
	output_ds = dict(zip(
		OUTPUT_DSNAMES,
		[data[dsname][:] for dsname in OUTPUT_DSNAMES]
	))

	renamed_output_ds = build_renamed(output_ds)
	
	splits = train_val_test_split(input_ds, renamed_output_ds, test_prop, val_prop)
	
	norm_output_ds = dict.fromkeys(['train_in', 'train_out', 'val_in', 'val_out', 'test_in', 'test_out'], None)
	
	# Normalize training targets and save statistics for identical processing on val and test sets
	norm_output_trainout, recons_trainout = normalize_outputs(
		splits['outputs'][0],
		zscore_norm = zscore_norm,
		minmax_norm = minmax_norm,
		recons = None,
		**kwargs,
	)
	norm_output_ds['train_in'] = splits['inputs'][0]['indexers/sequence']
	norm_output_ds['train_out'] = norm_output_trainout
	
	# Normalize (optional validation and) test targets using training statistics
	norm_output_testout, recons_testout = normalize_outputs(
		splits['outputs'][-1],
		zscore_norm = zscore_norm,
		minmax_norm = minmax_norm,
		recons = recons_trainout,
		**kwargs,
	)
	norm_output_ds['test_in'] = splits['inputs'][-1]['indexers/sequence']
	norm_output_ds['test_out'] = norm_output_testout
	
	if splits['inputs'][1] and splits['outputs'][1]:
		norm_output_val, recons_valout = normalize_outputs(
			splits['outputs'][1],
			zscore_norm = zscore_norm,
			minmax_norm = minmax_norm,
			recons = recons_trainout,
			**kwargs,
		)
		norm_output_ds['val_in'] = splits['inputs'][1]['indexers/sequence']
		norm_output_ds['val_out'] = norm_output_val
	
	return norm_output_ds


if __name__ == '__main__':
	import os
	
	dset_path = os.path.abspath(os.path.join('data', 'full_fixed_results.h5'))
	
	dataset = create_datasets(
		dset_path,
		zscore_norm = False,
		minmax_norm = True,
		test_prop = 0.2,
		val_prop = 0.2
	)
	
	for key, val in dataset.items():
		print(key)
		print(val)
		print()