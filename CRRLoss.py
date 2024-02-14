import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .Loss import Loss

class MRLoss(Loss):

	def __init__(self, adv_temperature = None,a = 0.0,b=0.1):
		super(MRLoss, self).__init__()
		self.a = nn.Parameter(torch.Tensor([a]))
		self.a.requires_grad = False
		self.b = nn.Parameter(torch.Tensor([b]))
		self.b.requires_grad = False
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False
	
	def get_weights(self, n_score):
		return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()
	

	def cliff_sigmod(self,x,a = 0.0,b=0.1):
		return 1/(1+torch.exp((a-x)/b))
	
	def forward(self, p_score, n_score):
		if self.adv_flag:
			return (self.get_weights(n_score) * self.cliff_sigmod(n_score-p_score,a = self.a, b = self.b)).sum(dim = -1).mean()
		else:
			return (torch.log(torch.sum(self.cliff_sigmod(n_score - p_score,a = self.a, b = self.b),axis = 1)+1.0)).mean()
	
	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()
	