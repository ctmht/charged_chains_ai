import os

from scipy.special import gammaln
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns


def logp_multinom(
	n_beads: int,
	n_aliph: int,
	n_anion: int
) -> np.float64:
	r"""
	Computes the density from the multinomial coefficient
	(n_beads choose n_aliph, n_aliph, n_anion, (n_beads - 2 * n_aliph - n_anion)).
	
	Args:
		n_beads (`int`): total chain length
		n_aliph (`int`): number of aliphatic monomers
		n_anion (`int`): number of anionic monomers
	"""
	return gammaln(n_beads + 1) - gammaln(n_aliph + 1) \
			- gammaln(n_beads - n_aliph - 2 * n_anion + 1) - 2 * gammaln(n_anion + 1)


def logp_reduced(
	n_beads: int,
	n_aliph: int,
	n_anion: int
) -> np.float64:
	r"""
	Compute the reduced probability of a macrostate given reversal symmetry.
	
	Args:
		n_beads (`int`): total chain length
		n_aliph (`int`): number of aliphatic monomers
		n_anion (`int`): number of anionic monomers
	"""
	log_pfull = logp_multinom(n_beads, n_aliph, n_anion)
	
	log_ppald = logp_multinom(n_beads // 2, n_aliph // 2, n_anion // 2)
	
	higher = log_pfull if log_pfull > log_ppald else log_ppald
	
	return higher + np.logaddexp(log_pfull - higher, log_ppald - higher)
	



def create_distribution_4nom(
	n_beads: int,
	temp: float = 1.0,
	apply_revsym: bool = True
) -> np.ndarray:
	r"""
	This function creates a file containing macrostate <-> probability mass pairs, given the
	proportion of each monomer in the sequence.
	
	We model sequences of fixed length n_beads containing types
		A: aliphatic
		B: aromatic
		C: anion
		D: cation
	Since n_anion = n_cation to ensure neutral charge, and the following n_beads - 2 * n_anion are either As or Bs,
	a tuple (n_aliph, n_anion) of aliphatic and anion numbers is sufficient to describe the macrostate at
	fixed total length n_beads. This is described by a multinomial coefficient. We compute this for each
	valid (n_aliph, n_anion) pair and normalize to obtain a 2D sample-able probability mass function.
	
	Args:
		n_beads (`int`): total length of the chain
		temp (`float`): temperature for distribution flattening (*)
		apply_revsym (`bool`): apply reversal symmetry to get the reduced distribution (**)
	
	(*) The multinomial coefficients are computed in logs for computational feasibility. They are
		converted back to floats by exponentiation. This results in unfathomly small numbers except
		for a single peak in the neighbourhood of (n_aliph, n_anion) ~= (25, 25). For better coverage
		during sampling, we define a temperature such that p = exp(logp / temp) instead (not
		counting the trivial distribution renormalization step at the end)
	
	(**) Assuming reversal symmetry for the sequences, one can get the smaller representative
		(lexicographically) and thus almost-halving the configuration space. However, this also
		requires accounting for the probability of palindromic sequences. 
	"""
	distribution = np.zeros(shape = (n_beads + 1, n_beads // 2 + 1), dtype = np.float64)
	
	
	for n_anion in range(0, n_beads // 2 + 1):
		n_cation = n_anion
		uncharged = n_beads - (n_anion + n_cation)
		for n_aliph in range(0, uncharged + 1):
			# n_aroma = uncharged - n_aliph
			
			if not apply_revsym:
				logphere = logp_multinom(n_beads, n_aliph, n_anion)
			elif n_anion % 2 == 0 and (n_beads % 2 != 0 or (n_beads % 2 == 0 and n_aliph % 2 == 0)):
				# There exist palindromic configurations for this (n_beads, n_aliph, n_anion)
				logphere = logp_reduced(n_beads, n_aliph, n_anion) - np.log(2)
			else:
				logphere = logp_multinom(n_beads, n_aliph, n_anion) - np.log(2)
			
			distribution[n_aliph, n_anion] = np.exp(logphere / temp)
	
	distribution /= np.sum(distribution)
	
	return distribution


def sample_distribution(
	distribution: np.ndarray,
	size: int = 1
) -> np.ndarray:
	r"""
	Samples the macrostate distribution
	"""
	samples_dist = np.zeros_like(distribution)
	dist_shape = distribution.shape
	
	samples_dist += np.bincount(
		np.random.choice(
			np.arange(0, dist_shape[0] * dist_shape[1], 1),
			p = distribution.flatten(),
			size = size
		),
		minlength = distribution.size
	).reshape(distribution.shape)
	
	return samples_dist


def get_sequences(
	samples_dist: np.ndarray,
	n_beads: int
) -> list[str]:
	r"""
	Creates as many microstates as the count of times a macrostate was sampled, by applying random
	permutations to strings of monomers with given macrostates. The function automatically
	implements and checks reversal symmetry and avoids repeated microstates.
	
	Args:
		samples_dist (`np.ndarray`):
		n_beads (`int`):
	
	Note: Written by DeepSeek, adjusted to fit style
	"""
	all_microstates = []
    
    # Get non-zero entries and their indices
	N_A_indices, N_C_indices = np.nonzero(samples_dist)
	
	for n_aliph, n_anion in zip(N_A_indices, N_C_indices):
		# Generate unique microstates for this (n_A, n_C)
		n_microstates = int(samples_dist[n_aliph, n_anion])
		microstates = generate_microstates(n_aliph, n_anion, n_microstates, n_beads)
		all_microstates.extend(microstates)
	
	return all_microstates


def generate_microstates(
	n_aliph: int,
	n_anion: int,
	n_micro: int,
	n_beads: int
) -> list[str]:
	"""
	Generate k unique microstates for a specific configuration
	
	Args:
		n_aliph (`int`):
		n_anion (`int`):
		n_micro (`int`):
		n_beads (`int`):
	
	Note: Written by DeepSeek, adjusted to fit style
	"""
	n_cation = n_anion
	n_aroma = n_beads - 2 * n_anion - n_aliph
	microstates = set()
	max_attempts = n_micro * 100  # Prevent infinite loops
	
	attempts = 0
	while len(microstates) < n_micro and attempts < max_attempts:
		# Generate random sequence with correct counts
		sequence = ['A'] * n_aliph + ['B'] * n_aroma + ['C'] * n_anion + ['D'] * n_cation
		np.random.shuffle(sequence)
		
		# Get canonical representative under reversal symmetry
		rep = ''.join(sequence)
		reprev = rep[: : -1]
		rep = rep if rep < reprev else reprev
		
		if rep not in microstates:
			microstates.add(rep)
		
		attempts += 1
	
	return list(microstates)



def plot_distribution(
	distribution,
	temp,
	ax
):
	r"""
	Create a heatmap of the macrostate distribution at a given temperature
	"""
	cmap = sns.cubehelix_palette(
		start = 0.5,
		rot = -0.5,
		dark = 0,
		light = .8,
		reverse = True,
		as_cmap = True
	) if temp == 1 else sns.mpl_palette(
		'hot',
		as_cmap = True
	)
	cmap.set_under(color='white')
	
	vticks = np.arange(0, distribution.shape[0], 10, dtype = int)
	hticks = np.arange(0, distribution.shape[1], 10, dtype = int)
	
	plotdistribution = np.copy(distribution)
	plotdistribution[distribution == 0] = -1
	
	ax = sns.heatmap(plotdistribution, cmap = cmap, annot = False, vmin = 0, ax = ax)
	ax.set_xticks(hticks, labels = hticks, rotation = 'horizontal')
	ax.set_yticks(vticks, labels = vticks, rotation = 'horizontal')
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	
	return ax
	

def plot_samples(
	distribution,
	samples_dist,
	temp,
	ax
):
	r"""
	Create a heatmap of samples from the macrostate distribution at a given temperature
	"""
	cmap = sns.cubehelix_palette(
		start = 0.5,
		rot = -0.5,
		dark = 0,
		light = .8,
		reverse = True,
		as_cmap = True
	) if temp == 1 else sns.mpl_palette(
		'hot',
		as_cmap = True
	)
	cmap.set_under(color='white')
	
	vticks = np.arange(0, distribution.shape[0], 10, dtype = int)
	hticks = np.arange(0, distribution.shape[1], 10, dtype = int)
	
	plotsampledistribution = np.copy(samples_dist)
	plotsampledistribution[distribution == 0] = -1
	
	ax = sns.heatmap(plotsampledistribution, cmap = cmap, annot = False, vmin = 0, ax = ax)
	ax.set_xticks(hticks, labels = hticks, rotation = 'horizontal')
	ax.set_yticks(vticks, labels = vticks, rotation = 'horizontal')
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	
	return ax


def plotting_checks():
	# Without symmetry
	fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 7), layout='constrained')
	
	temp = 1
	distribution = create_distribution_4nom(n_beads = 100, temp = temp, apply_revsym = False)
	samples_dist = sample_distribution(distribution, size = 10**3)
	axs[0, 0] = plot_distribution(distribution, temp, axs[0, 0])
	axs[1, 0] = plot_samples(distribution, samples_dist, temp, axs[1, 0])
	
	temp = 10
	distribution = create_distribution_4nom(n_beads = 100, temp = temp, apply_revsym = False)
	samples_dist = sample_distribution(distribution, size = 10**3)
	axs[0, 1] = plot_distribution(distribution, temp, axs[0, 1])
	axs[1, 1] = plot_samples(distribution, samples_dist, temp, axs[1, 1])
	
	axs[0, 0].set_xlabel('# anions ($N_C$)')
	axs[0, 1].set_xlabel('# anions ($N_C$)')
	axs[0, 0].set_ylabel('# aliphatic ($N_A$)')
	axs[1, 0].set_ylabel('# aliphatic ($N_A$)')
	
	for i in range(2):
		axs[1, i].set_xticks(ticks=[])
		axs[i, 1].set_yticks(ticks=[])
		for col in range(2): axs[i, col].grid(visible = False)
	
	fig.savefig('samples_without_revsym.pdf')
	plt.show()
	
	# With symmetry
	fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 7), layout='constrained')
	
	temp = 1
	distribution = create_distribution_4nom(n_beads = 100, temp = temp, apply_revsym = True)
	samples_dist = sample_distribution(distribution, size = 10**3)
	axs[0, 0] = plot_distribution(distribution, temp, axs[0, 0])
	axs[1, 0] = plot_samples(distribution, samples_dist, temp, axs[1, 0])
	
	temp = 10
	distribution = create_distribution_4nom(n_beads = 100, temp = temp, apply_revsym = True)
	samples_dist = sample_distribution(distribution, size = 10**3)
	axs[0, 1] = plot_distribution(distribution, temp, axs[0, 1])
	axs[1, 1] = plot_samples(distribution, samples_dist, temp, axs[1, 1])
	
	axs[0, 0].set_xlabel('# anions ($N_C$)')
	axs[0, 1].set_xlabel('# anions ($N_C$)')
	axs[0, 0].set_ylabel('# aliphatic ($N_A$)')
	axs[1, 0].set_ylabel('# aliphatic ($N_A$)')
	
	for i in range(2):
		axs[1, i].set_xticks(ticks=[])
		axs[i, 1].set_yticks(ticks=[])
		for col in range(2): axs[i, col].grid(visible = False)
	
	fig.savefig('samples_with_revsym.pdf')
	plt.show()
	
	microstates = get_sequences(samples_dist, n_beads = 100)
	# print(*microstates, sep='\n')


if __name__ == '__main__':
	DATA_FOLDER = os.path.abspath("./data/")
	
	USE_SYM = True
	TEMP = 10
	
	AUTOCORR_DATASET_SEED = 42
	AUTOCORR_DATASET_SIZE = 3 # 10 ** 3
	
	FULL_DATASET_SEED = 67
	FULL_DATASET_SIZE = 4 # 10 ** 4
	
	
	# Create the distribution of sequences, with reversal symmetry and temperature-based flattening
	distribution_revsym = create_distribution_4nom(n_beads = 100, temp = TEMP, apply_revsym = USE_SYM)
	
	# Save distribution in "data" folder
	filename = "distribution_revsym"
	filepath = os.path.join(DATA_FOLDER, filename)
	np.save(filepath, distribution_revsym)
	
	# Sample distribution and create unique sequences
	# 1. Autocorrelation analysis dataset
	np.random.seed(AUTOCORR_DATASET_SEED)
	samples_dist = sample_distribution(distribution_revsym, size = AUTOCORR_DATASET_SIZE)
	autocorr_sequences = get_sequences(samples_dist, n_beads = 100)
	
	filename = "sequences_autocorr_revsym.txt"
	filepath = os.path.join(DATA_FOLDER, filename)
	with open(filepath, "w") as autocorr_f:
		for seq in autocorr_sequences:
			autocorr_f.write(seq + "\n")
	
	# 2. Full dataset
	np.random.seed(FULL_DATASET_SEED)
	samples_dist = sample_distribution(distribution_revsym, size = FULL_DATASET_SIZE)
	full_sequences = get_sequences(samples_dist, n_beads = 100)
	
	filename = "sequences_full_revsym.txt"
	filepath = os.path.join(DATA_FOLDER, filename)
	with open(filepath, "w") as full_f:
		for seq in full_sequences:
			full_f.write(seq + "\n")