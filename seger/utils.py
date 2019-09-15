import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm 

import torch
import torchvision


def get_mean_std(dataset_loader):
    '''
    Computes mean and std

    Inputs:
    -------
    dataset_loader: torch.utils.data.DataLoader

    Outputs:
    -------
    mean : torch.tensor
    std : torch.tensor
    '''
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, _ in tqdm(dataset_loader):
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset_loader))
    std.div_(len(dataset_loader))
    print( mean, std)
    return mean,std


class AverageMeter:
	def __init__(self):
		self.sum = 0
		self.avg = 0
		self.samples = 0
		self.recentVal = 0

	def update(self, val):
		self.sum += val
		self.samples += 1
		self.recentVal = val
		self.avg = self.sum/self.samples



