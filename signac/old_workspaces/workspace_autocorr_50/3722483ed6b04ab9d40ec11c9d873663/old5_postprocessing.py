import post_processing as post

import pandas as pd
import torch

import os


def process():
	r"""
	
	"""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	topology = os.path.join(script_dir, '4_assembled.data')
	# trajectory = os.path.join(script_dir, '3_b_trajlin.data')
	trajectory = os.path.join(script_dir, '3_a_self_assemble.lammpstrj')
	universe = post.load(topology, trajectory)
	
	frame_times = []
	gyr_eigenvals = []
	end_to_end_vs = []
	ete_autocorrs = []
	
	radgyr_per_fr = []
	aspher_per_fr = []
	acylin_per_fr = []
	relsha_per_fr = []
	
	for frame in universe.trajectory:
		# Simulation time
		time = frame.time
		frame_times.append(time)
		
		positions = frame.positions
		
		# End-to-end distance vector and its autocorrelation
		end_to_end = post.end_to_end_distance(positions)
		end_to_end_vs.append(end_to_end)
		autocorr = post.end_to_end_autocorrelation(end_to_end_vs[0], end_to_end_vs[-1])
		ete_autocorrs.append(autocorr)
		
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
		frame_times = [frame_times],
		
		gyr_tensor_eigenvalues = [gyr_eigenvals],
		
		end_to_end_vectors = [end_to_end_vs],
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
	
	outfile = os.path.join(script_dir, 'final_output.csv')
	pd.DataFrame(results).to_csv(outfile)
	
	return ete_autocorrs

	


if __name__ == '__main__':
	# from importlib import resources
	# prototype_files_tvs = resources.files('simulation_prototype')
	# topology = prototype_files_tvs.joinpath('4_assembled.data')
	# trajectory = prototype_files_tvs.joinpath('3_b_trajlin.data')
	# universe = post.load(topology, trajectory)
	from matplotlib import pyplot as plt
	
	autocorrs = process()
	
	plt.plot(autocorrs[:100])
	
	plt.show()