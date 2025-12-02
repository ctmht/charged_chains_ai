from typing import Literal
import datetime
import os

from matplotlib import pyplot as plt
import pandas as pd
import torch


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



def analyze_autocorr_results(
	df: pd.DataFrame,
	data_folder: str,
	figures_folder: str,
):
	"""
	
	"""
	print("Columns:", df.columns)
	print("Head:", df.head(), sep='\n')
	
	# PLOT AUTOCORRELATION AND EXPONENTIAL DECAY FITS
	plt.figure(figsize=(13, 7), layout = 'constrained')
	
	squared = False
	log = True
	diff = True
	
	for idx, row in df.iterrows():
		autocorr = row["end_to_end_autocorrs"]
		times = torch.arange(start = 0, end = autocorr.shape[0], step = 1)
		
		tau = row["tau"] # Decay rate
		talpha = row["talpha_unconv"] # Where autocorrelation becomes insignificant (<5%) in the fit
		cutoff = row["cutoff"] # Where the empirical autocorrelation breaks monotonicity
		
		if squared: autocorr = torch.square(autocorr)
		factor = 2 if squared else 1
		
		if diff:
			plt.plot(times, autocorr - torch.exp(- factor/tau * times), color='k', lw=1,
		   			label="Autocorrelation-Fit difference" if idx==0 else None)
		else:
			plt.plot(times, autocorr, color='k', lw=1,
		   			label="Autocorrelation data" if idx==0 else None)
			plt.plot(times, torch.exp(- factor/tau * times), color='r', lw=1.5,
		   			label="Best exponential fit" if idx==0 else None)
		
		# plt.axvline(cutoff, ymin=-1, ymax=1, ls='--', lw=2, c='k',
		# 	label=f"$cutoff$: Monotonicity broken" if idx==0 else None)
		plt.axvline(talpha, ymin=-1, ymax=1, ls='-.', lw=1, c='green', zorder=-1,
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
	
	# plt.annotate("$atol = 10^{-5}$", xy=(150, -0.75), size=20, bbox=dict(boxstyle="square", fc="w"))
	
	# savepath = os.path.join(figures_folder, f'autocorr_{str(datetime.datetime.today()).split()[0]}_1e-5_diff.pdf')
	# plt.savefig(savepath)
	
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
		