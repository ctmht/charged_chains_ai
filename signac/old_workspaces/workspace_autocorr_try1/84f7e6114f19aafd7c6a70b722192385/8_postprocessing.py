from collections.abc import Callable
from typing import Literal
import sys
import os

# import freud  				# advanced analysis of molecular dynamics and other simulations
import MDAnalysis as mda  	# manipulate and analyse molecular dynamics trajectories
import pandas as pd
import numpy as np
import torch


###############################################################################
#############################    GENERAL UTILS    #############################
###############################################################################

#####################################################################
### Functions implementing general (statistical) methods
#####################################################################

def mean_covariance(
	elements: list[torch.Tensor]
) -> tuple[torch.Tensor]:
	r"""
	Computes mean vector and covariance matrix from a list of multivariate samples.
	
	Args:
		elements (`list[torch.Tensor]`): the list of samples to be used in the mean and
			covariance matrix computation. Iterable must contain `torch.Tensor` entries
	"""
	elements_arr = torch.stack(elements, dim = 0).mT
	
	mean = torch.mean(elements_arr, dim = 1)
	covariance = torch.cov(elements_arr)
	
	return mean, covariance


def expectation_variance(
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
	"""
	_jacobian_at_mean = torch.autograd.functional.jacobian(func, mean)
	_hessian_at_mean = torch.autograd.functional.hessian(func, mean)
	
	return (
		func(mean) + 0.5 * torch.trace(_hessian_at_mean @ cov),
		_jacobian_at_mean.transpose(0, -1) @ cov @ _jacobian_at_mean
	)


def sorted_eigenvalues(
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

def gyration_tensor(
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


def end_to_end_distance(
	atom_positions: np.ndarray
) -> torch.Tensor:
	r"""
	Compute the end-to-end distance (3-vector) given the list of all atom positions.
	
	Args:
		atom_positions (`np.ndarray`): position of atoms in the linear polymer chain. Must be a
			`np.ndarray`, and it is assumed that each row represents an individual atom (directly
			compatible with MDAnalysis frames)
	"""
	return torch.Tensor(atom_positions[0] - atom_positions[-1])


def end_to_end_autocorrelation(
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

def radius_of_gyration(
	gyr_evals_vec3: torch.Tensor
) -> float:
	r"""
	Compute the radius of gyration from gyration tensor eigenvalues:
		``R_g = sqrt(eval_1 ** 2 + eval_2 ** 2 + eval_3 ** 2) = ||gyr_evals_vec3||_2``
	
	Args:
		gyr_evals_vec3 (`torch.Tensor`): the three eigenvalues of the gyration tensor in a single
			vector. Eigenvalue ordering in this vector is irrelevant
	"""
	return torch.linalg.vector_norm(gyr_evals_vec3, ord = 2)


def asphericity(
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


def acylindricity(
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


def rel_shape_anisotropy(
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


def process_full_analysis():
	r"""
	# TODO: test that error propagation is indeed fine enough before a full simulation run
	# TODO: implement neighbourhood monomer proportions
	"""
	dt_integration = 0.005
	topology = '7_assembled.data'
	trajectory = '6_b_trajlin.data'
	universe = load(topology, trajectory, dt_integration)
	
	polymer_file = '1_polymer.mol'
	with open(polymer_file, 'r') as polf:
		_ = next(polf)
		seq = next(polf)[2:]
	
	outfile = '9_processed.pkl'
	
	gyr_eigenvals = []
	end_to_end_vs = []
	
	radgyr_per_fr = []
	aspher_per_fr = []
	acylin_per_fr = []
	relsha_per_fr = []
	
	for frame in universe.trajectory:
		positions = frame.positions
		
		# End-to-end distance vector and its autocorrelation
		end_to_end = end_to_end_distance(positions)
		end_to_end_vs.append(end_to_end)
		
		# Gyration tensor and its eigenvalues
		gyr_tensor = gyration_tensor(positions)
		gyr_eigenv = sorted_eigenvalues(gyr_tensor)
		gyr_eigenvals.append(gyr_eigenv)
		
		# Gyration tensor eigenvalue derived quantities, computed per-frame
		# TODO: If error propagation is appropriate then skip this
		rad_gyr = radius_of_gyration(gyr_eigenv)
		asphericity = asphericity(gyr_eigenv)
		acylindricity = acylindricity(gyr_eigenv)
		relshapeaniso = rel_shape_anisotropy(gyr_eigenv)
		
		radgyr_per_fr.append(rad_gyr)
		aspher_per_fr.append(asphericity)
		acylin_per_fr.append(acylindricity)
		relsha_per_fr.append(relshapeaniso)
	
	"""
	# Autocorrelation analysis
	maxoffset = 200
	times = torch.arange(0, maxoffset, 1)
	ete_autocorrs = end_to_end_autocorrelation(end_to_end_vs, maxoffset)
	tau, cutoff, talpha = fit_exponential_decay(X=times, Y=ete_autocorrs)
	"""
	
	"""
	# PLOT AUTOCORRELATION AND EXPONENTIAL DECAY FITS
	from matplotlib import pyplot as plt
	squared = True
	log = True
	if not squared:
		plt.plot(times, ete_autocorrs,
		   	color='blue', label="True data")
		plt.plot(times, torch.exp(-times/tau),
		   	color='magenta', label="Best fit to true data")
	else:
		plt.plot(times, torch.square(ete_autocorrs),
		   	color='blue', label="Squared data")
		plt.plot(times, torch.exp(- 2/tau * times),
		   	color='magenta', label="Best fit to squared data")
	if log and squared:
		plt.yscale('log')
	if not log:
		plt.plot(ete_autocorrs.diff(), ls='-', color='orange', label="Cumdiff of data")
	# plt.axhline(atol, xmin=0, xmax=200, ls='-', c='g', label="atol")
	plt.axvline(talpha, ymin=0, ymax=1, ls='-.', c='k', label="talpha")
	plt.axvline(cutoff, ymin=0, ymax=1, ls='--', c='r', label="cutoff")
	plt.legend()
	plt.show()
	"""
	
	# Iterative mean and variance
	radgyr_per_fr = torch.as_tensor(radgyr_per_fr)
	radgyr_per_fr_mean = torch.mean(radgyr_per_fr)
	radgyr_per_fr_var = torch.var(radgyr_per_fr)
	aspher_per_fr = torch.as_tensor(aspher_per_fr)
	aspher_per_fr_mean = torch.mean(aspher_per_fr)
	aspher_per_fr_var = torch.var(aspher_per_fr)
	acylin_per_fr = torch.as_tensor(acylin_per_fr)
	acylin_per_fr_mean = torch.mean(acylin_per_fr)
	acylin_per_fr_var = torch.var(acylin_per_fr)
	relsha_per_fr = torch.as_tensor(relsha_per_fr)
	relsha_per_fr_mean = torch.mean(relsha_per_fr)
	relsha_per_fr_var = torch.var(relsha_per_fr)
	
	# Mean and covariance matrix of (decreasingly) ordered eigenvalues
	mean, cov = mean_covariance(gyr_eigenvals)
	
	# Gyration tensor eigenvalue derived quantities, computed by error propagation
	prop_rad_gyr = expectation_variance(radius_of_gyration, mean, cov)
	prop_asphericity = expectation_variance(asphericity, mean, cov)
	prop_acylindricity = expectation_variance(acylindricity, mean, cov)
	prop_relshapeaniso = expectation_variance(rel_shape_anisotropy, mean, cov)
	
	results = dict(
		sequence = [seq],
		
		maxtime = [universe.trajectory[-1].time],
		
		gyr_tensor_eigenvalues = [torch.stack(gyr_eigenvals, dim = 0)],
		
		end_to_end_vectors = [torch.stack(end_to_end_vs, dim = 0)],
		# end_to_end_autocorrelations = [ete_autocorrs],
		
		mean_eigenvalues = [mean],
		covariance_eigenvalues = [cov],
		
		R_g_perfm = [radgyr_per_fr_mean],
		R_g_perfv = [radgyr_per_fr_var],
		R_g_propm = [prop_rad_gyr[0]],
		R_g_propv = [prop_rad_gyr[1]],
		b_perfm = [aspher_per_fr_mean],
		b_perfv = [aspher_per_fr_var],
		b_propm = [prop_asphericity[0]],
		b_propv = [prop_asphericity[1]],
		c_perfm = [acylin_per_fr_mean],
		c_perfv = [acylin_per_fr_var],
		c_propm = [prop_acylindricity[0]],
		c_propv = [prop_acylindricity[1]],
		kappa2_perfm = [relsha_per_fr_mean],
		kappa2_perfv = [relsha_per_fr_var],
		kappa2_propm = [prop_relshapeaniso[0]],
		kappa2_propv = [prop_relshapeaniso[1]]
	)
	
	df = pd.DataFrame(results)
	
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
	for frame in universe.trajectory:
		positions = frame.positions
		
		end_to_end = end_to_end_distance(positions)
		end_to_end_vs.append(end_to_end)
	
	# Autocorrelation analysis
	maxoffset = 200
	times = torch.arange(0, maxoffset, 1)
	ete_autocorrs = end_to_end_autocorrelation(end_to_end_vs, maxoffset)
	tau, cutoff, talpha = fit_exponential_decay(X=times, Y=ete_autocorrs)
	
	# Convert the determined talpha to actual simulation time and number of steps
	talpha_st = talpha * (universe.trajectory[1].time - universe.trajectory[0].time)
	talpha_ss = talpha_st / dt_integration
	
	# Saving results
	results = dict(
		sequence = [seq],
		end_to_end = [end_to_end_vs],
		maxoffset_autocorrs = [maxoffset],
		end_to_end_autocorrs = [ete_autocorrs],
		tau = [tau],
		cutoff = [cutoff],
		talpha_unconv = [talpha],
		talpha_simtime = [talpha_st],
		talpha_simsteps = [talpha_ss]
	)
	df = pd.DataFrame(results)
	df.to_pickle(outfile)
	
	# TODO: remove on actual large simulations
	outfile = '9_processed_DUMP.csv'
	df.to_csv(outfile)
	
	"""
	# PLOT AUTOCORRELATION AND EXPONENTIAL DECAY FITS
	from matplotlib import pyplot as plt
	plt.figure(figsize=(8, 3), layout = 'constrained')
	squared = True
	log = True
	if not squared:
		plt.plot(times, ete_autocorrs,
		   	color='blue', label="True data")
		plt.plot(times, torch.exp(-times/tau),
		   	color='magenta', label="Best fit to true data")
	else:
		plt.plot(times, torch.square(ete_autocorrs),
		   	color='blue', label="Squared data")
		plt.plot(times, torch.exp(- 2/tau * times),
		   	color='magenta', label="Best fit to squared data")
	if log and squared:
		plt.yscale('log')
	if not log:
		plt.plot(ete_autocorrs.diff(), ls='-', color='orange', label="Cumdiff of data")
	# plt.axhline(atol, xmin=0, xmax=200, ls='-', c='g', label="atol")
	plt.axvline(talpha, ymin=0, ymax=1, ls='-.', c='k', label=f"$T_\\alpha = {talpha}$ $(\\alpha = 0.95)$")
	plt.axvline(cutoff, ymin=0, ymax=1, ls='--', c='r', label=f"$cutoff = {cutoff}$")
	plt.legend()
	plt.xlim([0, 100])
	plt.ylim([0, torch.exp(- 2/tau * 0)])
	plt.xlabel('Simulation dump step (every $1000dt$)')
	plt.ylabel('Autocorrelation')
	plt.savefig('autocorr_sqlog.pdf')
	plt.show()
	"""

	


if __name__ == '__main__':
	taskname: Literal["autocorr", "full"] = sys.argv[1]
	print(taskname)
	
	if taskname == "autocorr":
		process_autocorrelation()
	else:
		process_full_analysis()