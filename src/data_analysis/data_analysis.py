from typing import Literal
import datetime
import os

from matplotlib import pyplot as plt
import pandas as pd
import torch


def analyze_autocorr_results(
	df: pd.DataFrame,
	data_folder: str,
	figures_folder: str,
):
	"""
	
	"""
	print(df.columns)
	
	print(df.head())
	
	# PLOT AUTOCORRELATION AND EXPONENTIAL DECAY FITS
	plt.figure(figsize=(13, 7), layout = 'constrained')
	
	squared = False
	log = True
	
	for idx, row in df.iterrows():
		autocorr = row["end_to_end_autocorrs"]
		if squared: autocorr = torch.square(autocorr)
		
		times = torch.arange(start = 0, end = autocorr.shape[0], step = 1)
		
		tau = row["tau"] # Decay rate
		talpha = row["talpha_unconv"] # Where autocorrelation becomes insignificant (<5%) in the fit
		cutoff = row["cutoff"] # Where the empirical autocorrelation breaks monotonicity
		
		factor = 2 if squared else 1
		
		plt.plot(times, autocorr, color='k', lw=1,
		   label="Autocorrelation data" if idx==0 else None)
		plt.axvline(cutoff, ymin=-1, ymax=1, ls='--', lw=2, c='k',
			  label=f"$cutoff$: Monotonicity broken" if idx==0 else None)
		
		plt.plot(times, torch.exp(- factor/tau * times), color='r', lw=1.5,
		   label="Best exponential fit" if idx==0 else None)
		plt.axvline(talpha, ymin=-1, ymax=1, ls='-.', lw=2.5, c='r',
			  label=f"$T_\\alpha$ $(\\alpha = 0.95)$: insignificance threshold" if idx==0 else None)
		
		# if log and squared:
		# 	plt.yscale('log')
		# if not log:
		# 	plt.plot(ete_autocorrs.diff(), ls='-', color='orange', label="Cumdiff of data")
		
		# plt.axhline(atol, xmin=0, xmax=200, ls='-', c='g', label="atol")
	
	plt.legend(loc = "upper right", fontsize = 13)
	plt.xlim([0, 200])
	plt.ylim([-1, 1])
	plt.xlabel('Number of autocorrelation lag steps ($1$ step every $1000dt$)', size = 15)
	plt.ylabel('Autocorrelation', size = 15)
	
	savepath = os.path.join(figures_folder, f'autocorr_{str(datetime.datetime.today()).split()[0]}.pdf')
	plt.savefig(savepath)
	
	plt.show()


def analyze_full_results(
	df: pd.DataFrame,
	data_folder: str,
	figures_folder: str,
):
	...


if __name__ == '__main__':
	DATA_FOLDER = os.path.abspath("./data/")
	FIGURES_FOLDER = os.path.abspath("./figures/")
	
	for taskname in ["autocorr", "full"]:
		respath = os.path.join(DATA_FOLDER, f"{taskname}_results.pkl")
		
		if os.path.exists(respath):
			df = pd.read_pickle(respath)
			
			if taskname == "autocorr":
				analyze_autocorr_results(df, data_folder=DATA_FOLDER, figures_folder=FIGURES_FOLDER)
			else:
				analyze_full_results(df, data_folder=DATA_FOLDER, figures_folder=FIGURES_FOLDER)
		