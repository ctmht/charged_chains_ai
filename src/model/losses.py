from typing import Optional, Literal

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import torch.nn as nn
import torch


class NLL(nn.Module):
	""" Implementation of the negative log likelihood for Gaussian RVs """

	def __init__(
		self,
		**kwargs
	) -> None:
		"""
		
		"""
		super().__init__()
	
	
	def forward(
		self,
		mean: torch.Tensor,
		cov: torch.Tensor,
		target: torch.Tensor,
		batch_reduction: Literal["mean", "sum", "none"] = "none"
	) -> torch.Tensor:
		"""
		
		"""
		dist = MultivariateNormal(loc = mean, covariance_matrix = cov)
		nll = -dist.log_prob(target)
		
		if batch_reduction == "sum":
			return nll.sum(dim = 0)
		elif batch_reduction == "mean":
			return nll.mean(dim = 0)
		else:
			return nll



class KLDivergence(nn.Module):
	""" KL Divergence for Gaussian distributions """
	
	def __init__(
		self,
		**kwargs
	) -> None:
		"""
		
		"""
		super().__init__()
	
	def forward(
		self,
		pred_mean: torch.Tensor,
		pred_cov: torch.Tensor,
		true_mean: torch.Tensor,
		true_cov: torch.Tensor,
		batch_reduction: Literal["mean", "sum", "none"] = "none"
	) -> torch.Tensor:
		"""
			
		"""
		# Create multivariate normal distributions
		pred_dist = MultivariateNormal(pred_mean, pred_cov)
		true_dist = MultivariateNormal(true_mean, true_cov)
		
		# Compute KL divergence
		kl = kl_divergence(true_dist, pred_dist)
		
		if batch_reduction == "sum":
			return kl.sum(dim = 0)
		elif batch_reduction == "mean":
			return kl.mean(dim = 0)
		else:
			return kl


if __name__ == '__main__':
	BATCH_SIZE = 2
	mean_true = torch.rand(BATCH_SIZE, 3)
	mean_pred = torch.rand(BATCH_SIZE, 3)
	
	cov_true = torch.rand(BATCH_SIZE, 3, 3)
	cov_true = torch.einsum('bji, bjk -> bik', cov_true, cov_true)[0]
	cov_pred = torch.rand(BATCH_SIZE, 3, 3)
	cov_pred = torch.einsum('bji, bjk -> bik', cov_pred, cov_pred)[0]
	
	cov_pred = torch.diag_embed(torch.rand(BATCH_SIZE, 3))
	
	print(mean_true, cov_true, mean_pred, cov_pred, sep='\n')
	
	kldivloss = KLDivergence()
	print(kldivloss(mean_pred, cov_pred, mean_true, cov_true, batch_reduction='none'))
	
	nllloss = NLL()
	print(nllloss(mean_pred, cov_pred, mean_true, batch_reduction='none'))