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


def correlation(
	sample1,
	sample2
) -> torch.Tensor:
	r"""
	Computes correlation of two identically-sized random variables with zero mean (e.g. the
	(auto)correlation of the end-to-end distance vector between the first and current time steps)
	
	TODO: this is not okay
	
	Args:
		items (`list` or iterable): per-frame attribute to be used in autocorrelation computation
	"""
	_sample1 = torch.as_tensor(sample1)
	_sample2 = torch.as_tensor(sample2)
	
	_corr = (_sample1.transpose(0, -1) @ _sample2) / (_sample1.transpose(0, -1) @ _sample1)
	
	return _corr
	


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
	return torch.linalg.vector_norm(gyr_evals_vec3)


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