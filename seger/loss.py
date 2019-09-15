""" Networks to train deep neural networks
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def get_loss(loss, kwargs):

    weights = kwargs["weights"]
    class_weights = torch.FloatTensor(weights).cuda()
    
    if loss == "dice":
        print("Dice Loss multi class")
        return GeneralizedDiceLoss(kwargs["num_classes"])
    elif loss == "ce_loss":
        return torch.nn.CrossEntropyLoss()
    elif loss == "weighed_ce_loss":
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise NotImplementedError("This is not implemented yet.. Raise pull request")
        
        
def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)
        
class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=False):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        
        #convert target into one-hot encoded form
        target_1_hot = torch.eye(2)[target.squeeze(1)]
        target_1_hot = target_1_hot.permute(0, 3, 1, 2).float()
        target = target_1_hot

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            input = input * mask
            target = target * mask

        input = flatten(input)
        target = flatten(target)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.epsilon), requires_grad=False)

#         class_weights = torch.FloatTensor([4.0356e-13, 1.0088e-09])
        
        target= target.cuda()
        class_weights = class_weights.cuda()
        
        #print(class_weights)

        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)


       
