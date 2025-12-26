from collections.abc import Callable
from collections import Counter
from typing import Literal, Any
import sys
import os

# import freud  				# advanced analysis of molecular dynamics and other simulations
import MDAnalysis as mda  	# manipulate and analyse molecular dynamics trajectories
import lammps_logfile
import pandas as pd
import numpy as np
import scipy as sp
import torch


###############################################################################
#############################    GENERAL UTILS    #############################
###############################################################################

#####################################################################
### Functions implementing sequence descriptors (blockiness)
#####################################################################

def get_blockinesses(
	sequence: str
) -> dict[Literal['A', 'B', 'C', 'D'], float]:
	"""
	Computes a measure of the sequence blockiness based on the nontrivial (!= 1) eigenvalue
	of a 2 x 2 Markov transition matrix, for sequences containing two bead types. If the
	sequence contains more bead types (e.g. ABCD for this project), the function creates a
	4 x 4 Markov transition matrix: for each type, the rows and columns are collapsed as if
	representing a chain with two types (X and notX), and the eigenvalue is computed as usual
	
	Args:
		sequence (`str`): The monomer sequence defining the linear polymer chain. By default,
			the function supports strings containing [A, B, C, D] or nonempty subsets thereof,
			other behaviours are not checked
	Returns:
		blockinesses (`dict[Literal['A', 'B', 'C', 'D'], float]`): Dictionary of bead types (keys)
			and associated blockiness parameters (the nontrivial eigenvalue of the 2x2 matrix)
	"""
	# Count length-2 subsequences
	monomers = sorted(list(Counter(sequence).keys()))
	
	counts = {ch1 + '1': {ch2 + '2': 0 for ch2 in monomers} for ch1 in monomers}
    
	for idx in range(len(sequence) - 1):
		ch1 = sequence[idx]
		ch2 = sequence[idx + 1]
		counts[ch1 + '1'][ch2 + '2'] += 1
    
	# Create transition matrix and make (row-)stochastic
	mat = pd.DataFrame(counts, dtype = float).apply(lambda row: row / row.sum()).T.values
	if mat.sum() == len(sequence) - 1:
		mat /= (len(sequence) - 1)
	mat /= mat.sum(axis = 1)
	
	blockinesses: dict = {}
	
	if len(monomers) <= 2:
		# The matrix is already trivial (2 x 2 for 2 monomer types)
		# The nontrivial eigenvalue is = trace - 1
		blockinesses[monomers[0]] = np.trace(mat) - 1
		return blockinesses
	
	for coll_target in monomers:
		# Collapse non-targets and keep target
		idx = monomers.index(coll_target)
		
		YY = mat[idx, idx]
		YN = mat[idx, :].sum() - mat[idx, idx]
		NY = mat[:, idx].sum() - mat[idx, idx]
		NN = mat.sum() - YY - YN - NY
		NY /= 3
		NN /= 3
		
		collapsed = np.array([
			[YY, YN],
			[NY, NN]
		])
		blockinesses[coll_target] = np.trace(collapsed) - 1
	
	return blockinesses


#####################################################################
### Functions implementing general (statistical) methods
#####################################################################

def get_mean_covariance(
	elements: list[torch.Tensor]
) -> tuple[torch.Tensor]:
	r"""
	Computes mean vector and covariance matrix from a list of multivariate samples.
	
	Args:
		elements (`list[torch.Tensor]`): the list of samples to be used in the mean and
			covariance matrix computation. Iterable must contain `torch.Tensor` entries
	Returns:
		mean, covariance (`tuple[torch.Tensor]`): the mean and covariance matrix obtained
			from the given list
	"""
	elements_arr = torch.stack(elements, dim = 0).mT
	
	mean = torch.mean(elements_arr, dim = 1)
	covariance = torch.cov(elements_arr, correction = 1)
	
	return mean, covariance


def get_expectation_variance(
	func: Callable[[torch.Tensor], float],
	mean: torch.Tensor,
	cov: torch.Tensor
) -> tuple[float, float]:
	r"""
	Computes the expectation and variance of a scalar-valued vector function `f` by the delta
	method for error propagation using the Taylor expansion of `f`:
	- for expectation `E[f(X)]`, the second order approximation is used;
	- for variance `V[f(X)]`, the first order approximation is used.
	
	This makes use of the mean vector and the covariance matrix of the input in the computation
	and `torch.autograd` for automatic differentiation of the function `f`.
	
	The function returns the pair of scalars `(E[f(X)], V[f(X)])`.
	
	Args:
		func (`Callable`): the scalar-valued vector function `f` to be approximated
		mean (`torch.Tensor`): the mean vector `E[X]`
		cov (`torch.Tensor`): the covariance matrix `V[X]`
	Returns:
		(expectation, variance) (`tuple[float, float]`): the expectation and variance computed
			using delta method for error propagation
	"""
	_jacobian_at_mean = torch.autograd.functional.jacobian(func, mean)
	_hessian_at_mean = torch.autograd.functional.hessian(func, mean)
	
	return (
		func(mean) + 0.5 * torch.trace(_hessian_at_mean @ cov),
		_jacobian_at_mean.transpose(0, -1) @ cov @ _jacobian_at_mean
	)


def get_sorted_eigenvalues(
	symmetric_matrix: torch.Tensor
) -> torch.Tensor:
	r"""
	Get eigenvalues of a symmetric matrix (e.g. gyration tensor) and sort in descending order
	
	Args:
		symmetric_matrix (`torch.Tensor`): symmetric matrix
	"""
	return torch.linalg.eigvalsh(symmetric_matrix).flip(dims = (-1,))


#####################################################################
### Task-specific frame processing functions
#####################################################################

def get_gyration_tensor(
	atom_positions: np.ndarray
) -> torch.Tensor:
	r"""
	Compute the gyration tensor from atom positions. Function assumes that the positions are in
	matrix form, where each atom position is a row (compatible with MDAnalysis frames)
	
	Args:
		atom_positions (`np.ndarray`): position of atoms in the linear polymer chain. Must be a
			`np.ndarray`, and it is assumed that each row represents an individual atom (directly
			compatible with MDAnalysis frames)
	"""
	_atom_positions = torch.as_tensor(atom_positions)
	_center = torch.mean(_atom_positions, dim = 0)
	_positions_centered = _atom_positions - _center
	_n_atoms = _atom_positions.shape[0]
	
	return _positions_centered.mT @ _positions_centered / _n_atoms


def get_end_to_end_vector(
	atom_positions: np.ndarray
) -> torch.Tensor:
	r"""
	Compute the end-to-end distance (3-vector) given the list of all atom positions.
	
	Args:
		atom_positions (`np.ndarray`): position of atoms in the linear polymer chain. Must be a
			`np.ndarray`, and it is assumed that each row represents an individual atom (directly
			compatible with MDAnalysis frames)
	Returns:
		end-to-end vector (`torch.Tensor`)
	"""
	return torch.Tensor(atom_positions[0] - atom_positions[-1])


def get_end_to_end_autocorrelation(
	end_to_end_vs: torch.Tensor,
	lag_max: int
) -> torch.Tensor:
	r"""
	
	"""
	end_to_end_vs_t = torch.stack(end_to_end_vs, dim = 1)
	n_samples = end_to_end_vs_t.shape[1]
	
	autocorr = torch.zeros(lag_max)
	
	for t_lag in range(lag_max):
		series0 = end_to_end_vs_t[:, 0 : n_samples - t_lag]
		series0_sum = series0.sum(dim = 1)
		series0_sum_norm = torch.linalg.vector_norm(series0_sum, ord = 2) ** 2
		series0_norm_sum = torch.trace(series0.transpose(0, -1) @ series0)
		series0_std = torch.sqrt((n_samples - t_lag) * series0_norm_sum - series0_sum_norm)
		
		seriest = end_to_end_vs_t[:, t_lag : n_samples]
		seriest_sum = seriest.sum(dim = 1)
		seriest_sum_norm = torch.linalg.vector_norm(seriest_sum, ord = 2) ** 2
		seriest_norm_sum = torch.trace(seriest.transpose(0, -1) @ seriest)
		seriest_std = torch.sqrt((n_samples - t_lag) * seriest_norm_sum - seriest_sum_norm)
		
		series0t_inner = torch.trace(series0.transpose(0, -1) @ seriest)
		
		numerator = (n_samples - t_lag) * series0t_inner - series0_sum.transpose(0, -1) @ seriest_sum
		denominator = series0_std * seriest_std
		
		autocorr[t_lag] = numerator / denominator
	
	return autocorr


def fit_exponential_decay(
	X: torch.Tensor,
	Y: torch.Tensor,
	alpha: float = 0.95,
	atol: float = 1e-3
) -> tuple[float, int]:
	r"""
	Fits an exponential decay of the form Y = exp(- X / k) (*) and determines cutoff point X0 where
	at most alpha of the density lies in the region X >= X0.
	
	Args:
		X (`torch.Tensor`): the x values of the data which we want to fit to
		Y (`torch.Tensor`): the y values of the data
		alpha (`float`): confidence level for determining cutoff after fitting the regression. This
			happens through the use of the exponential distribution quantile function once the rate
			parameter k has been determined. Note that alpha=1.00 coresponds to the point at X=+inf
			and alpha=0.00 corresponds to X=0, both unappropriate for any practical purpose
		atol (`float`): absolute tolerance for breaking monotonicity. Since the use of this
			function targets autocorrelation functions, it is expected that this decreases 'nicely'
			before starting to fluctuate uncontrollably at an undetermined cutoff point. A high
			tolerance will result in the entire data being used in the regression, but a too low
			tolerance might cut off the data too early to provide an accurate regression line.
	Returns:
		tau:
		cutoff:
		talpha:
	
	(*) To be precise, the function fits the regression ln(Y**2) = (-2 / k) X in order to find k,
		a transformation we perform to ensure the logarithm is well-defined.
	"""
	if not 0 < alpha < 1:
		raise ValueError(f"'alpha' confidence level must be between 0 and 1, got {alpha=}")
	if atol < 0:
		raise ValueError(f"'atol' absolute tolerance must be non-negative, got {atol=}")
	
	# Determine cutoff for "well-behaved" part of autocorrelation
	# return index of *first* position where this is true
	cutoff = torch.argmax((Y.diff() > atol).int())
	cutoff = Y.shape[0] if cutoff == 0 else cutoff
	
	# Linear regression without intercept: ln(Y^2) = (-2/tau) X
	# Analytical solution: tau = - (sum_i (2 T[i])**2) / (sum_i (2 T[i] ln(Y^2[i])))
	Yp2 = torch.square(Y[:cutoff])
	Xt2 = 2 * X[:cutoff]
	tau = - torch.sum(torch.square(Xt2)) / torch.sum(Xt2 * torch.log(Yp2))
	
	# Determine X-value such that beyond there is less than (1-alpha) autocorrelation "density"
	# use the quantile function (inverse CDF) of exponential distribution
	icdf = - torch.log(torch.scalar_tensor(1 - alpha)) * tau
	t_alpha = torch.ceil(icdf)
	
	return tau, cutoff, t_alpha


#####################################################################
### Simulation statistics specific to gyration tensor eigenvalues
#####################################################################

def get_radius_of_gyration_sq(
	gyr_evals_vec3: torch.Tensor
) -> float:
	r"""
	Compute the (!squared) radius of gyration from gyration tensor eigenvalues:
		``R_g = sqrt(eval_1 ** 2 + eval_2 ** 2 + eval_3 ** 2) = ||gyr_evals_vec3||_2``
	
	Args:
		gyr_evals_vec3 (`torch.Tensor`): the three eigenvalues of the gyration tensor in a single
			vector. Eigenvalue ordering in this vector is irrelevant
	"""
	return torch.linalg.vector_norm(gyr_evals_vec3, ord = 2)


def get_asphericity(
	gyr_evals_vec3: torch.Tensor
) -> float:
	r"""
	Compute the asphericity from (ordered) gyration tensor eigenvalues:
		``b = eval_1 ** 2 - 0.5 * (eval_2 ** 2 + eval_3 ** 2)``
	
	Args:
		gyr_evals_vec3 (`torch.Tensor`): the three eigenvalues of the gyration tensor in a single
			vector. Eigenvalues in this vector *must* be sorted in decreasing order, since this is
			not enforced internally (`gyr_evals_vec3[0] > gyr_evals_vec3[1] > gyr_evals_vec3[2]`)
	"""
	return gyr_evals_vec3[0] ** 2 - 0.5 * (gyr_evals_vec3[1] ** 2 + gyr_evals_vec3[2] ** 2)


def get_acylindricity(
	gyr_evals_vec3: torch.Tensor
) -> float:
	r"""
	Compute the asphericity from (ordered) gyration tensor eigenvalues:
		``c = eval_2 ** 2 - eval_3 ** 2``
	
	Args:
		gyr_evals_vec3 (`torch.Tensor`): the three eigenvalues of the gyration tensor in a single
			vector. Eigenvalues in this vector *must* be sorted in decreasing order, since this is
			not enforced internally (`gyr_evals_vec3[0] > gyr_evals_vec3[1] > gyr_evals_vec3[2]`)
	"""
	return gyr_evals_vec3[1] ** 2 - gyr_evals_vec3[2] ** 2


def get_rel_shape_anisotropy(
	gyr_evals_vec3: torch.Tensor
) -> float:
	r"""
	Compute the relative shape anisotropy from gyration tensor eigenvalues:
		``kappa ** 2 = 1.5 * (||gyr_evals_vec3||_4 ** 4 / ||gyr_evals_vec3||_2 ** 4) - 0.5 ``
	
	Args:
		gyr_evals_vec3 (`torch.Tensor`): the three eigenvalues of the gyration tensor in a single
			vector. Eigenvalue ordering in this vector is irrelevant
	"""
	_norm2to4 = torch.linalg.vector_norm(gyr_evals_vec3, ord = 2) ** 4
	_norm4to4 = torch.linalg.vector_norm(gyr_evals_vec3, ord = 4) ** 4
	return 1.5 * (_norm4to4 / _norm2to4) - 0.5


def get_neighbour_distribution(
	sequence: str,
	atom_positions: np.ndarray,
	count_bonded: bool = False,
	cutoff: float = 2,
	mean: bool = True
):
	"""
	Get the (mean) count of neighbours of each of the four types (A, B, C, D) around each of
	the N (= 100) beads in the linear chain.
	
	Args:
		atom_positions (`numpy.ndarray`): The array of 100 x 3 bead positions in a given frame
			of the simulation for which this computation should be done
		count_bonded (`bool`): By default, this count does not include the beads which are bonded
			to be neighbours (False)
		cutoff (`float`): The distance at which to cut off the count of neighbouring beads,
			expected in LJ units. By default, each bead looks within a distance of 2 of itself
		mean (`bool`): If False, the full 100 x 4 neighbour count is returned. When True, the
			function will aggregate using the mean to return a 4-element vector of mean neighbour
			counts per-type over the entire chain
	"""
	# Compute distance map of the linear chain
	distance_map = sp.spatial.distance.squareform(
		sp.spatial.distance.pdist(atom_positions)
	)
	
	# For each bead, remove itself from the counts
	diag_indices = np.arange(0, distance_map.shape[0], 1, dtype=int)
	distance_map[diag_indices, diag_indices] = np.inf
	
	if not count_bonded:
		# For each bead, remove the bounded neighbours (linear => first before and after itself)
		offsubdiag_indices = np.arange(0, distance_map.shape[0] - 1, 1, dtype=int)
		distance_map[offsubdiag_indices, offsubdiag_indices + 1] = np.inf
		distance_map[offsubdiag_indices + 1, offsubdiag_indices] = np.inf
	
	# Clip distance map at cutoff distance
	contact_map = distance_map <= cutoff
	
	# Encode bead types in the order they appear in the sequence
	type_codes = np.array([ord(c) for c in sequence]) - ord('A')
	one_hot = np.zeros((len(sequence), 4), dtype = int)
	one_hot[np.arange(len(sequence)), type_codes] = 1
	# Count neighbours per type using one-hot encoding over contact map
	counts_unnorm = contact_map.astype(int) @ one_hot
	
	if mean:
		# Average per-type neighbour counts over all beads
		return np.mean(counts_unnorm, axis = 0)
	
	# Return all counts
	return counts_unnorm



def load(
	topology: str,
	trajectory: str,
	dt_integration = 0.005
) -> mda.Universe:
	r"""
	Create a MDAnalysis Universe based on the given (final) molecule topology and its trajectory
	"""
	return mda.Universe(topology, trajectory, format="LAMMPSDUMP", dt=dt_integration)


###############################################################################
#########################  POST-PROCESSING FUNCTIONS  #########################
###############################################################################

def process_full_analysis(
	job_folder: str | os.Path
):
	r"""
	
	"""
	dt_integration = 0.005

	polymer_file = os.path.join(job_folder, '1_polymer.mol')
	log_file = os.path.join(job_folder, '6_a_log.lammps')
	trajectory = os.path.join(job_folder, '6_b_trajlin.data')
	topology = os.path.join(job_folder, '7_assembled.data')
	outfile = os.path.join(job_folder, '9_processed.pkl')

	# Get molecule sequence
	with open(polymer_file, 'r') as polf:
		_ = next(polf)
		sequence = next(polf)[2:]
	sequence = sequence[:-1] if sequence[-1] == '\n' else sequence
	
	# Get monomer counts (A and C suffice here)
	counts = Counter(sequence)
	count_aliphatic_A = counts['A']
	count_anion_C = counts['C']
	
	# Get the four blockiness parameters
	blockinesses = get_blockinesses(sequence)
	for montype in 'ABCD':
		if montype not in blockinesses:
			blockinesses[montype] = 0
	
	# Potential energy getter
	# -> There are two runs in the LAMMPS simulation, first (1) for minimization and second
	# -> (2) full, NOT zero-indexed in the lammps_logfile library
	logged_data = lammps_logfile.File(log_file)
	poteng = logged_data.get("PotEng", run_num = 2)
	poteng_perfm = np.mean(poteng)
	poteng_perfs = np.std(poteng, ddof = 1)
	
	# Load universe
	universe = load(topology, trajectory, dt_integration)
	
	
	gyr_tensor_oevals = []
	radgyr2_perf = []
	aspher_perf = []
	acylin_perf = []
	relsha_perf = []
	
	for frame in universe.trajectory:
		positions = frame.positions
		
		# Gyration tensor and its (ordered/sorted) eigenvalues
		gyr_tensor = get_gyration_tensor(positions)
		oevals = get_sorted_eigenvalues(gyr_tensor)
		gyr_tensor_oevals.append(oevals)
		
		# Gyration tensor eigenvalue derived quantities, computed per-frame
		# TODO: remove after errprop check
		radgyr2 = get_radius_of_gyration_sq(oevals)
		radgyr2_perf.append(radgyr2)
		
		aspher = get_asphericity(oevals)
		aspher_perf.append(aspher)
		
		acylin = get_acylindricity(oevals)
		acylin_perf.append(acylin)
		
		relsha = get_rel_shape_anisotropy(oevals)
		relsha_perf.append(relsha)
	
	# Per-frame means and variances
	# TODO: remove after errprop check
	radgyr2_perf = torch.as_tensor(radgyr2_perf).detach().numpy()
	radgyr2_perfm = np.mean(radgyr2_perf)
	radgyr2_perfs = np.std(radgyr2_perf, ddof = 1)
	
	aspher_perf = torch.as_tensor(aspher_perf).detach().numpy()
	aspher_perfm = np.mean(aspher_perf)
	aspher_perfs = np.std(aspher_perf, ddof = 1)
	
	acylin_perf = torch.as_tensor(acylin_perf).detach().numpy()
	acylin_perfm = np.mean(acylin_perf)
	acylin_perfs = np.std(acylin_perf, ddof = 1)
	
	relsha_perf = torch.as_tensor(relsha_perf).detach().numpy()
	relsha_perfm = np.mean(relsha_perf)
	relsha_perfs = np.std(relsha_perf, ddof = 1)
	
	# Mean and covariance matrix of (decreasingly) ordered eigenvalues
	gyr_tensor_oevals_perfm, gyr_tensor_oevals_perfv = get_mean_covariance(gyr_tensor_oevals)
	
	# Gyration tensor eigenvalue derived quantities, computed by error propagation
	radgyr2_propm, radgyr2_propv = get_expectation_variance(get_radius_of_gyration_sq,
		gyr_tensor_oevals_perfm, gyr_tensor_oevals_perfv)
	radgyr2_props = torch.sqrt(radgyr2_propv).detach().numpy()
	
	aspher_propm, aspher_propv = get_expectation_variance(get_asphericity,
		gyr_tensor_oevals_perfm, gyr_tensor_oevals_perfv)
	aspher_props = torch.sqrt(aspher_propv).detach().numpy()
	
	acylin_propm, acylin_propv = get_expectation_variance(get_acylindricity,
		gyr_tensor_oevals_perfm, gyr_tensor_oevals_perfv)
	acylin_props = torch.sqrt(acylin_propv).detach().numpy()
	
	relsha_propm, relsha_propv = get_expectation_variance(get_rel_shape_anisotropy,
		gyr_tensor_oevals_perfm, gyr_tensor_oevals_perfm)
	relsha_props = torch.sqrt(relsha_propv).detach().numpy()
	
	# Get neighbour counts at last dump frame
	positions = universe.trajectory[-1].positions
	nb_mean = get_neighbour_distribution(
		sequence,
		positions,
		count_bonded = False,
		cutoff = 2,
		mean = True
	)
	nb_mean_dict = dict(zip('ABCD', nb_mean))
	
	# Define results
	results: dict[str, Any] = {
		"job_id": os.path.basename(job_folder),
		# Sequence and descriptors
		"sequence": sequence,
		"count_aliphatic_A": count_aliphatic_A,
		"count_anion_C": count_anion_C,
		"blockiness_A": blockinesses['A'],
		"blockiness_B": blockinesses['B'],
		"blockiness_C": blockinesses['C'],
		"blockiness_D": blockinesses['D'],
		# Shape and descriptors
		## Gyration tensor - ordered eigenvalues (with vector mean and covariance matrix)
		# "gyr_tensor_oevals": gyr_tensor_oevals, # TODO: remove since might be superfluous
		"gyr_tensor_oevals_perfm": gyr_tensor_oevals_perfm,
		"gyr_tensor_oevals_perfv": gyr_tensor_oevals_perfv,
		## Squared radius of gyration R_g^2
		"radgyr2_perfm": radgyr2_perfm,
		"radgyr2_perfs": radgyr2_perfs,
		"radgyr2_propm": radgyr2_propm,
		"radgyr2_props": radgyr2_props,
		## Asphericity b
		"aspher_perfm": aspher_perfm,
		"aspher_perfs": aspher_perfs,
		"aspher_propm": aspher_propm,
		"aspher_props": aspher_props,
		## Acylindricity c
		"acylin_perfm": acylin_perfm,
		"acylin_perfs": acylin_perfs,
		"acylin_propm": acylin_propm,
		"acylin_props": acylin_props,
		## Relative shape anisotropy \kappa^2
		"relsha_perfm": relsha_perfm,
		"relsha_perfs": relsha_perfs,
		"relsha_propm": relsha_propm,
		"relsha_props": relsha_props,
		## Potential energy V
		"poteng_perfm": poteng_perfm,
		"poteng_perfs": poteng_perfs,
		## Neighbourhoods
		"nbc_perbm_A": nb_mean_dict['A'],
		"nbc_perbm_B": nb_mean_dict['B'],
		"nbc_perbm_C": nb_mean_dict['C'],
		"nbc_perbm_D": nb_mean_dict['D']
		
	}
	
	# Save results
	df = pd.DataFrame(pd.Series(results)).T
	df.to_pickle(outfile)



def process_autocorrelation():
	"""
	
	"""
	# Define inputs and load universe
	dt_integration = 0.005
	topology = '7_assembled.data'
	trajectory = '6_b_trajlin.data'
	universe = load(topology, trajectory, dt_integration)
	
	# Extract monomer sequence from molecule file
	polymer_file = '1_polymer.mol'
	with open(polymer_file, 'r') as polf:
		_ = next(polf)
		seq = next(polf)[2:]
	
	
	# Define output file
	outfile = '9_processed.pkl'
	
	# Get end-to-end distance vector at each frame
	end_to_end_vs = []
	
	printed = False
	for frame in universe.trajectory:
		if not printed:
			print(f"{frame}\n{frame.time=}\n{frame.dt=}")
			printed = True
		
		positions = frame.positions
		
		end_to_end = get_end_to_end_vector(positions)
		end_to_end_vs.append(end_to_end)
	
	# Parameters
	maxlag = 200   		# number of lags to compute autocorrelation for
	alpha = 0.95		# quantile for cutoff in the exponential fit
	atol = 1e-5			# absolute tolerance for nonmonotonicity
	
	# Autocorrelation analysis
	lags = torch.arange(0, maxlag, 1)
	ete_autocorrs = get_end_to_end_autocorrelation(end_to_end_vs, maxlag)
	decay_factor, nonmon_cutoff, quantile_cutoff = fit_exponential_decay(
		X = lags, Y = ete_autocorrs,
		alpha = 0.95, atol = 1e-5
	)
	
	# Convert the determined lag_cutoff to actual simulation time and number of steps
	timediff = universe.trajectory[1].time - universe.trajectory[0].time
	lagstep_cutoff_simtime = quantile_cutoff * timediff
	lagstep_cutoff_simsteps = lagstep_cutoff_simtime / dt_integration
	
	# Saving results
	results = dict(
		sequence = [seq],
		end_to_end = [end_to_end_vs],
		maxoffset_autocorrs = [maxlag],
		end_to_end_autocorrs = [ete_autocorrs],
		atol = [atol],
		alpha = [alpha],
		tau = [decay_factor],
		cutoff = [nonmon_cutoff],
		talpha_unconv = [quantile_cutoff],
		talpha_simtime = [lagstep_cutoff_simtime],
		talpha_simsteps = [lagstep_cutoff_simsteps]
	)
	df = pd.DataFrame(results)
	df.to_pickle(outfile)


if __name__ == '__main__':
	taskname = "_TEST"
	if len(sys.argv) > 1:
		taskname: Literal["autocorr", "full"] = sys.argv[1]
		print("Run script with task", taskname)
	
	match taskname:
		case "autocorr":
			process_autocorrelation()
		case "full":
			process_full_analysis()
		case "_TEST" | _:
			...