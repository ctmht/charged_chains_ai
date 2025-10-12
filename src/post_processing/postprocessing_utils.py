import freud  				# advanced analysis of molecular dynamics and other simulations
import MDAnalysis as mda  	# manipulate and analyse molecular dynamics trajectories
import numpy as np
import torch

from collections.abc import Callable


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




def load_old():
	# TODO: update, refactor
	folder = 'src/'
	topology = folder+'final.data'
	trajectory_lin = folder+'traj.lin'
	trajectory_log = folder+'traj.log'
	
	# Time step size used in the MD simulations (in reduced units)
	dt_integration = 0.005
	
	# Creating Universe objects for linear and logarithmic trajectories with MDAnalysis
	# These objects represent the entire system and allow for analysis of the data
	universe_lin = mda.Universe(topology, trajectory_lin, format='LAMMPSDUMP', dt=dt_integration)
	universe_log = mda.Universe(topology, trajectory_log, format='LAMMPSDUMP', dt=dt_integration)
	
	# Calculating the number of frames and beads (particles) in the simulations
	n_frames_log = universe_log.trajectory.n_frames  	# Total number of frames in the logarithmic trajectory
	n_frames_lin = universe_lin.trajectory.n_frames  	# Total number of frames in the linear trajectory
	n_beads = universe_lin.atoms.n_atoms  			 	# Total number of beads (atoms) in the system
	
	results = {
		'time': [],
		'gte': [],
		'rg': [],
		'rsa_sq': [],
		'ete': [],
		'b': [],
		'c': []
	}
	
	for frame in universe_lin.trajectory:
		time = frame.time
		end_to_end_distance = np.linalg.norm(frame.positions[0] - frame.positions[-1])
		
		min_positions = torch.min(torch.as_tensor(frame.positions), dim = 0).values
		max_positions = torch.max(torch.as_tensor(frame.positions), dim = 0).values
		bound_distance = torch.linalg.norm(max_positions - min_positions)
		
		gyr_tensor = get_gyration_tensor(frame.positions)
		gyr_eigenvalues = get_sorted_eigenvalues(gyr_tensor)
		rad_gyr = torch.sqrt(torch.sum(gyr_eigenvalues))
		
		results['time'].append(time)
		results['ete'].append(end_to_end_distance)
		results['gte'].append(gyr_eigenvalues)
		
		results['rg'].append(rg(gyr_eigenvalues))
		results['b'].append(b(gyr_eigenvalues))
		results['c'].append(c(gyr_eigenvalues))
		results['rsa_sq'].append(rsa(gyr_eigenvalues))
		
		print(f"{frame.time : >6.0f}, {end_to_end_distance : >12.6f}, {rad_gyr : >9.6f}, {bound_distance : >12.6f},  {gyr_eigenvalues}")
	print(f" (time)  (end to end)  (rad gyr)  (bound dist)  (gyr tensor evalues)")
	
	#################################################################
	## PyTorch KL Divergence between two multivariate normals
	#################################################################
	
	mean, covariance = get_mean_covariance(results['gte'])
	
	print(
		"\nFROM REGULAR DISTRIBUTION\n",
		'    mean', mean,
		'    covariance', covariance,
		'    sorted evalues', get_sorted_eigenvalues(covariance),
		sep='\n'
	)
	
	dist1 = torch.distributions.MultivariateNormal(
		mean,
		covariance
	)
	dist2 = torch.distributions.MultivariateNormal(
		torch.Tensor([0, 0, 0]),
		torch.eye(3)
	)
	dist3 = torch.distributions.MultivariateNormal(
		torch.Tensor([30, 10, 5]),
		covariance
	)
	
	print(
		'\nFarther dist:', torch.distributions.kl.kl_divergence(dist1, dist2).item(),
		'\nCloser dist:', torch.distributions.kl.kl_divergence(dist1, dist3).item(),
		'\n'
	)
	
	#################################################################
	## Testing mean R_g over time versus R_g defined by mean evalues
	#################################################################
	arr_pf = torch.as_tensor(results['rg'])
	print(f"R_g   from value at each time step: {torch.mean(arr_pf).item() : >12.5f} +- {torch.std(arr_pf).item() : >12.5f}")
	tarmean, tarstd = get_radgyr_with_std(mean, covariance)
	print(f"R_g  from mean+-cov of eigenvalues: {tarmean.item() : >12.5f} +- {tarstd.item() : >12.5f}")
	
	arr_pf = torch.as_tensor(results['b'])
	print(f"  b   from value at each time step: {torch.mean(arr_pf).item() : >12.5f} +- {torch.std(arr_pf).item() : >12.5f}")
	tarmean, tarstd = get_b_with_std(mean, covariance)
	print(f"  b  from mean+-cov of eigenvalues: {tarmean.item() : >12.5f} +- {tarstd.item() : >12.5f}")
	
	arr_pf = torch.as_tensor(results['c'])
	print(f"  c   from value at each time step: {torch.mean(arr_pf).item() : >12.5f} +- {torch.std(arr_pf).item() : >12.5f}")
	tarmean, tarstd = get_c_with_std(mean, covariance)
	print(f"  c  from mean+-cov of eigenvalues: {tarmean.item() : >12.5f} +- {tarstd.item() : >12.5f}")
	
	arr_pf = torch.as_tensor(results['rsa_sq'])
	print(f"k^2   from value at each time step: {torch.mean(arr_pf).item() : >12.5f} +- {torch.std(arr_pf).item() : >12.5f}")
	tarmean, tarstd = get_relshapeaniso_with_std(mean, covariance)
	print(f"k^2  from mean+-cov of eigenvalues: {tarmean.item() : >12.5f} +- {tarstd.item() : >12.5f}")
	

if __name__ == '__main__':
	...
	# load()
	
	# n_frames = 6
	# dim = 3
	# ete = [np.random.rand(dim)**2 for _ in range(n_frames)]
	
	# print(ete)
	
	# autocorr = [correlation(ete[0], ete[i]) for i in range(n_frames)]
	# autocorr = torch.Tensor(autocorr)
	# print(autocorr)