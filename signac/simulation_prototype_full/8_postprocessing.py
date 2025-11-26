import post_processing as post

import pandas as pd
import torch

import os


def process():
	r"""
	
	"""
	dt_integration = 0.005
	script_dir = os.path.dirname(os.path.abspath(__file__))
	topology = os.path.join(script_dir, '7_assembled.data')
	trajectory = os.path.join(script_dir, '6_b_trajlin.data')
	# trajectory = os.path.join(script_dir, '6_a_self_assemble.lammpstrj')
	universe = post.load(topology, trajectory, dt_integration)
	
	polymer_file = os.path.join(script_dir, '1_polymer.mol')
	with open(polymer_file, 'r') as polf:
		_ = next(polf)
		seq = next(polf)[2:]
	
	# run_file = os.path.join(script_dir, '4_LAMMPS_mnr.in')
	# with open(run_file, 'r') as runf:
	# 	while True:
	# 		line = next(runf)
	# 		if 'subsample_rate' in line: break
	# 	subsample_rate = int(line.split(' ')[-1])
	
	outfile = os.path.join(script_dir, '9_processed.pkl')
	
	gyr_eigenvals = []
	end_to_end_vs = []
	
	radgyr_per_fr = []
	aspher_per_fr = []
	acylin_per_fr = []
	relsha_per_fr = []
	
	for frame in universe.trajectory:
		positions = frame.positions
		
		# End-to-end distance vector and its autocorrelation
		end_to_end = post.end_to_end_distance(positions)
		end_to_end_vs.append(end_to_end)
		
		# Gyration tensor and its eigenvalues
		gyr_tensor = post.gyration_tensor(positions)
		gyr_eigenv = post.sorted_eigenvalues(gyr_tensor)
		gyr_eigenvals.append(gyr_eigenv)
		
		# Gyration tensor eigenvalue derived quantities, computed per-frame
		# TODO: If error propagation is appropriate then skip this
		rad_gyr = post.radius_of_gyration(gyr_eigenv)
		asphericity = post.asphericity(gyr_eigenv)
		acylindricity = post.acylindricity(gyr_eigenv)
		relshapeaniso = post.rel_shape_anisotropy(gyr_eigenv)
		
		radgyr_per_fr.append(rad_gyr)
		aspher_per_fr.append(asphericity)
		acylin_per_fr.append(acylindricity)
		relsha_per_fr.append(relshapeaniso)
	
	# Autocorrelation analysis
	maxoffset = 200
	times = torch.arange(0, maxoffset, 1)
	ete_autocorrs = post.end_to_end_autocorrelation(end_to_end_vs, maxoffset)
	tau, cutoff, talpha = post.fit_exponential_decay(X=times, Y=ete_autocorrs)
	
	
	
	
	
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
	mean, cov = post.mean_covariance(gyr_eigenvals)
	
	# Gyration tensor eigenvalue derived quantities, computed by error propagation
	prop_rad_gyr = post.expectation_variance(post.radius_of_gyration, mean, cov)
	prop_asphericity = post.expectation_variance(post.asphericity, mean, cov)
	prop_acylindricity = post.expectation_variance(post.acylindricity, mean, cov)
	prop_relshapeaniso = post.expectation_variance(post.rel_shape_anisotropy, mean, cov)
	
	results = dict(
		sequence = [seq],
		
		maxtime = [universe.trajectory[-1].time],
		
		gyr_tensor_eigenvalues = [torch.stack(gyr_eigenvals, dim = 0)],
		
		end_to_end_vectors = [torch.stack(end_to_end_vs, dim = 0)],
		end_to_end_autocorrelations = [ete_autocorrs],
		
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
	script_dir = os.path.dirname(os.path.abspath(__file__))
	topology = os.path.join(script_dir, '7_assembled.data')
	trajectory = os.path.join(script_dir, '6_b_trajlin.data')
	universe = post.load(topology, trajectory, dt_integration)
	
	# Extract monomer sequence from molecule file
	polymer_file = os.path.join(script_dir, '1_polymer.mol')
	with open(polymer_file, 'r') as polf:
		_ = next(polf)
		seq = next(polf)[2:]
	
	# # Extract subsampling rate from LAMMPS input file
	# run_file = os.path.join(script_dir, '4_LAMMPS_mnr.in')
	# with open(run_file, 'r') as runf:
	# 	while True:
	# 		line = next(runf)
	# 		if 'subsample_rate' in line: break
	# 	subsample_rate = int(line.split(' ')[-1])
	
	# Define output file
	outfile = os.path.join(script_dir, '9_processed.pkl')
	
	# Get end-to-end distance vector at each frame
	end_to_end_vs = []
	for frame in universe.trajectory:
		positions = frame.positions
		
		end_to_end = post.end_to_end_distance(positions)
		end_to_end_vs.append(end_to_end)
	
	# Autocorrelation analysis
	maxoffset = 200
	times = torch.arange(0, maxoffset, 1)
	ete_autocorrs = post.end_to_end_autocorrelation(end_to_end_vs, maxoffset)
	tau, cutoff, talpha = post.fit_exponential_decay(X=times, Y=ete_autocorrs)
	
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
	outfile = os.path.join(script_dir, '9_processed_DUMP.csv')
	df.to_csv(outfile)
	
	
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

	


if __name__ == '__main__':
	process()