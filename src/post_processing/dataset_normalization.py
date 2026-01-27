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
	newmax: float = 1.0
) -> tuple[np.ndarray, float, float]:
	oldmin = series.min(axis = 0)
	oldmax = series.max(axis = 0)
	minmax_norm_series = (series - oldmin) / (oldmax - oldmin) * newmax + newmin
	
	return minmax_norm_series, oldmin, oldmax


def zscore_normalize(
	series: np.ndarray
) -> tuple[np.ndarray, float, float]:
	oldmean = series.mean(axis = 0)
	oldstd = series.std(axis = 0, ddof = 1)
	zscore_norm_series = (series - oldmean) / oldstd

	return zscore_norm_series, oldmean, oldstd


import numpy as np

'''
def normalize_spd_cholesky(covariances, normalizer, eps=1e-12, *args):
	"""
	Normalize SPD matrices via Cholesky decomposition.

	Args:
		covariances: (N, d, d) array of SPD matrices
		target_min, target_max: Target range for normalized elements
		eps: Small constant for numerical stability
		
	Returns:
		normalized_cov: (N, d, d) normalized SPD matrices
		stats: Dict with normalization parameters AND normalized Cholesky factors
	"""
	N, d, _ = covariances.shape

	# 1. Regularize for numerical stability
	reg_cov = covariances + eps * np.eye(d)[np.newaxis, :, :]

	# 2. Cholesky decomposition
	L = np.linalg.cholesky(reg_cov)  # Shape: (N, d, d)

	# 3. Extract lower triangular elements
	tril_idx = np.tril_indices(d)
	L_elements = L[:, tril_idx[0], tril_idx[1]]  # Shape: (N, n_elements)

	# 4. Compute statistics and normalize
	L_min = L_elements.min(axis=0, keepdims=True)
	L_max = L_elements.max(axis=0, keepdims=True)

	# Avoid division by zero
	L_range = L_max - L_min
	L_range[L_range == 0] = 1.0

	L_norm_elements, olda, oldb = normalizer(L_elements, *args)
	# L_norm_elements = (L_elements - L_min) / L_range
	# L_norm_elements = L_norm_elements * (target_max - target_min) + target_min

	# 5. Store normalized elements in stats (CRITICAL FIX)
	stats = {
		'L_mean' if normalizer is zscore_normalize else 'L_min': olda,
		'L_std' if normalizer is zscore_normalize else 'L_max': oldb,
		'tril_idx': tril_idx,
		'L_norm_elements': L_norm_elements  # Store normalized elements
	}
	
	# 6. Reconstruct normalized Cholesky factors
	L_norm = np.zeros_like(L)
	L_norm[:, tril_idx[0], tril_idx[1]] = L_norm_elements

	# 7. Reconstruct normalized SPD matrices
	normalized_cov = L_norm @ L_norm.transpose(0, 2, 1)

	return normalized_cov, stats


def denormalize_spd_cholesky(stats, eps=1e-12):
    """
    Denormalize using stored normalized Cholesky factors.
    
    Args:
        stats: Dict from normalize_spd_cholesky
        eps: Small constant (not used here, kept for compatibility)
        
    Returns:
        original_scale_cov: (N, d, d) SPD matrices at original scale
    """
    # Extract stored elements and parameters
    L_norm_elements = stats['L_norm_elements']
    L_min = stats['L_min']
    L_max = stats['L_max']
    target_min = stats['target_min']
    target_max = stats['target_max']
    tril_idx = stats['tril_idx']
    
    N = L_norm_elements.shape[0]
    d = len(tril_idx[0])  # Number of unique elements in lower triangle
    matrix_dim = int((np.sqrt(8*d + 1) - 1) / 2)  # Solve n*(n+1)/2 = d
    
    # 1. Reverse the normalization of Cholesky elements
    L_range = L_max - L_min
    L_range[L_range == 0] = 1.0
    
    L_orig_elements = (L_norm_elements - target_min) / (target_max - target_min)
    L_orig_elements = L_orig_elements * L_range + L_min
    
    # 2. Reconstruct original Cholesky factors
    L_orig = np.zeros((N, matrix_dim, matrix_dim))
    L_orig[:, tril_idx[0], tril_idx[1]] = L_orig_elements
    
    # 3. Reconstruct original SPD matrices
    original_scale_cov = L_orig @ L_orig.transpose(0, 2, 1)
    
    return original_scale_cov
'''

def normalize_outputs(
	output_dsets: dict[str, np.ndarray],
	zscore_norm: bool = True,
	minmax_norm: bool = True,
	**kwargs
) -> tuple[dict[str, np.ndarray] | dict[str, np.ndarray | None]]:
	"""
	Applies normalization to each entry

	If both zscore and minmax are True, then zscore is always applied first
	"""
	outputs_zsn = {}
	
	if zscore_norm:
		oldmeans = {}
		oldstds = {}
		for key in output_dsets:
			if 'gte_oev' in key:
				outputs_mmn[key] = output_dsets[key]
				oldmeans[key] = None
				oldstds[key] = None
			else:
				outputs_zsn[key], oldmeans[key], oldstds[key] = zscore_normalize(output_dsets[key])
	else:
		for key in output_dsets:
			outputs_zsn[key] = output_dsets[key]
	
	outputs_mmn = {}
	if minmax_norm:
		oldmins = {}
		oldmaxs = {}
		# oldtriudxm = {}
		for key in output_dsets:
			if 'gte_oev' in key:
				outputs_mmn[key] = output_dsets[key]
				oldmins[key] = None
				oldmaxs[key] = None
			else:
				outputs_mmn[key], oldmins[key], oldmaxs[key] = minmax_normalize(outputs_zsn[key], **kwargs)
	else:
		for key in output_dsets:
			outputs_mmn[key] = outputs_zsn[key]

	reconstructors = {
		'oldmins': oldmins if minmax_norm else None,
		'oldmaxs': oldmaxs if minmax_norm else None,
		'oldmeans': oldmeans if zscore_norm else None,
		'oldstds': oldstds if zscore_norm else None,
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
		'inputs': (train_input_dsets, test_input_dsets),
		'outputs': (train_output_dsets, test_output_dsets)
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
	**kwargs
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
	
	print(renamed_output_ds['nbc'].shape)
	
	norm_output_ds, recons = normalize_outputs(renamed_output_ds, zscore_norm=zscore_norm, minmax_norm=minmax_norm, **kwargs)
	
	splits = train_val_test_split(input_ds, norm_output_ds, test_prop, val_prop)
	
	return {
		'train_in': splits['inputs'][0]['indexers/sequence'],
		'train_out': splits['outputs'][0],
		'val_in': splits['inputs'][1]['indexers/sequence'],
		'val_out': splits['outputs'][1],
		'test_in': splits['inputs'][2]['indexers/sequence'],
		'test_out': splits['outputs'][2],
		'reconstruct': recons
	}