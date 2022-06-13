from argparse import Namespace
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from model.base import BaseSelfSupervisedModel

class SupConLoss(nn.Module):
	"""Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
	It also supports the unsupervised contrastive loss in SimCLR"""
	def __init__(self, temperature=5, contrast_mode='all',
					base_temperature=5):
		super(SupConLoss, self).__init__()
		self.temperature = temperature
		self.contrast_mode = contrast_mode
		self.base_temperature = base_temperature

	def forward(self, features, labels=None, mask=None):
		"""Compute loss for model. If both `labels` and `mask` are None,
		it degenerates to SimCLR unsupervised loss:
		https://arxiv.org/pdf/2002.05709.pdf
		Args:
			features: hidden vector of shape [bsz, n_views, ...].
			labels: ground truth of shape [bsz].
			mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
				has the same class as sample i. Can be asymmetric.
		Returns:
			A loss scalar.
		"""
		device = torch.device('cuda')

		# features : (N, n_views, f_dim)
		# n_views : 2, the num of crops in an image

		batch_size = features.shape[0]
		labels = labels.contiguous().view(-1, 1) # (N, 1)
		mask = torch.eq(labels, labels.T).float().to(device) # label 동일 여부에 따라 1 or 0

		contrast_count = features.shape[1] # n_views==2
		contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # (2N, f_dim)
		if self.contrast_mode == 'one':
			anchor_feature = features[:, 0]
			anchor_count = 1
		elif self.contrast_mode == 'all':
			anchor_feature = contrast_feature # (2N, f_dim)
			anchor_count = contrast_count # 2
		else:
			raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

		# 코드를 왜 굳이 이렇게 지저분하게 짰지..?
		# compute logits
		anchor_dot_contrast = torch.div(
			torch.matmul(anchor_feature, contrast_feature.T), # (2N, 2N)
			self.temperature)
		# for numerical stability 
		logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
		logits = anchor_dot_contrast - logits_max.detach()

		# tile mask
		mask = mask.repeat(anchor_count, contrast_count) # diag, upper diag, lower diag (2N, 2N)
		# mask-out self-contrast cases
		logits_mask = torch.scatter( # (parameter로 입력한) 주어진 index에 맞춰 src값을 새로운 tesnor로 할당 
			torch.ones_like(mask), # input, (2N, 2N)
			1, # dim 
			torch.arange(batch_size * anchor_count).view(-1, 1).to(device), # index, (0,1,2,3,4,5,6,7)
			0 # src
		) # 그냥 대각 성분만 인 행렬이네.. 
		mask = mask * logits_mask # Positive가 1인,,

		# compute log_prob
		exp_logits = torch.exp(logits) * logits_mask
		log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # 나눗셈을 이따구로한거임 지금...

		# compute mean of log-likelihood over positive
		# summation over all positives
		mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) # 분모: |P(i)|

		# loss
		# summation over all samples
		loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
		loss = loss.view(anchor_count, batch_size).mean()

		return loss