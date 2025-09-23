import freud  				# advanced analysis of molecular dynamics and other simulations
import MDAnalysis as mda  	# manipulate and analyse molecular dynamics trajectories
import numpy as np
import torch

from scipy.differentiate import jacobian as gradient, hessian

from collections.abc import Callable


def get_gyration_tensor(atom_positions: np.ndarray) -> torch.Tensor:
	"""
	Compute the gyration tensor from atom positions. Function assumes that the positions are in
	matrix form, where each atom position is a row (due to how MDAnalysis frames work)
	"""
	atom_positions = torch.as_tensor(atom_positions)
	center = torch.mean(atom_positions, dim = 0)
	positions_centered = atom_positions - center
	
	n_atoms = atom_positions.shape[0]
	return positions_centered.mT @ positions_centered / n_atoms


def get_sorted_eigenvalues(symmetric_matrix: torch.Tensor) -> torch.Tensor:
	"""
	Get eigenvalues of a symmetric matrix (e.g. gyration tensor) and sort in descending order
	"""
	return torch.linalg.eigvalsh(symmetric_matrix).flip(dims = (-1,))


def get_mean_covariance(elements: list[torch.Tensor]) -> tuple[torch.Tensor]:
	"""
	Get mean vector and covariance matrix from a list of multivariate samples
	"""
	elements_arr = torch.stack(elements, dim = 0).mT
	
	mean = torch.mean(elements_arr, dim = 1)
	covariance = torch.cov(elements_arr)
	
	return mean, covariance




def expectation(
	func: Callable[[torch.Tensor], float],
	mean: torch.Tensor,
	cov: torch.Tensor
) -> float:
	return func(mean) + 0.5 * torch.trace(
		torch.autograd.functional.hessian(func, mean) @ cov
	)
	
def variance(
	func: Callable[[torch.Tensor], float],
	mean: torch.Tensor,
	cov: torch.Tensor
) -> float:
	return (
		torch.autograd.functional.jacobian(func, mean).transpose(0, -1)
		@ cov @
		torch.autograd.functional.jacobian(func, mean)
	)


def rg(vec3: torch.Tensor) -> float:
	return torch.linalg.vector_norm(vec3)
def b(vec3: torch.Tensor) -> float:
	# Lz^2 - 0.5(Lx^2 + Ly^2)   Lx <= Ly <= Lz
	# Lx^2 - 0.5(Ly^2 + Lz^2)   Lx >= Ly >= Lz
	return 1.5 * vec3[0] ** 2 - 0.5 * (vec3[1] ** 2 + vec3[2] ** 2)
def c(vec3: torch.Tensor) -> float:
	# Ly^2 - Lx^2               Lx <= Ly <= Lz
	# Ly^2 - Lz^2               Lx >= Ly >= Lz
	return vec3[1] ** 2 - vec3[2] ** 2
def rsa(vec3: torch.Tensor) -> float:
	# 1.5 norm4^4 / norm2^4 - 0.5
	return 1.5 * torch.linalg.vector_norm(vec3, ord = 4) ** 4 / torch.linalg.vector_norm(vec3, ord = 2) ** 4 - 0.5

def get_radgyr_with_std(
	mean: torch.Tensor,
	cov: torch.Tensor
) -> tuple[float]:
	"""
	Get the radius of gyration R_g +- stddev from the final eigenvector mean and covariance
	matrix using the delta method for error propagation (and expectation estimation)
	"""
	return expectation(rg, mean, cov), torch.sqrt(variance(rg, mean, cov))

def get_relshapeaniso_with_std(
	mean: torch.Tensor,
	cov: torch.Tensor
) -> tuple[float]:
	"""
	
	"""
	return expectation(rsa, mean, cov), torch.sqrt(variance(rsa, mean, cov))

def get_b_with_std(
	mean: torch.Tensor,
	cov: torch.Tensor
) -> tuple[float]:
	"""
	
	"""
	return expectation(b, mean, cov), torch.sqrt(variance(b, mean, cov))

def get_c_with_std(
	mean: torch.Tensor,
	cov: torch.Tensor
) -> tuple[float]:
	"""
	
	"""
	return expectation(c, mean, cov), torch.sqrt(variance(c, mean, cov))


def load():
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
	load()