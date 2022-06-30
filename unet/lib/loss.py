import torch
import numpy as np
import torch.nn.functional as F

from . import utils

def BCELogitsLoss(y_hat, y, weight = None):
    return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=weight)

def BCEDiceLoss(y_hat, y, weight = 0.1, device = 'cuda'):
    bce_loss = F.binary_cross_entropy_with_logits(y_hat, y)
    y_hat = torch.sigmoid(y_hat) 
    
    _, dice_loss = utils.dice_coeff_batch(y_hat, y, device)
    loss = bce_loss * weight + dice_loss * (1 - weight)

    return loss

def DiceLoss(y_hat, y):
    y_hat = torch.sigmoid(y_hat) 
    _, dice_loss = utils.dice_coeff_batch(y_hat, y)
    return dice_loss

def FocalLoss(y_hat, y, alpha=1, gamma=2, logits=True, reduce=True):
    if logits:
        BCE_loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none')
    else:
        BCE_loss = F.binary_cross_entropy(y_hat, y, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss

    if reduce:
        return torch.mean(F_loss)
    else:
        return F_loss


# def NLLLoss(y_hat, y):
#     # loss = torch.nn.NLLLoss()
#     # m = torch.nn.LogSoftmax(dim=1) # assuming tensor is of size N x C x height x width, where N is the batch size.
#     loss = F.nll_loss(F.log_softmax(y_hat), y)
#     return loss

# class WeightedFocalLoss(torch.nn.Module):
#     "Non weighted version of Focal Loss"
#     "https://amaarora.github.io/2020/06/29/FocalLoss.html"
#     def __init__(self, alpha=.25, gamma=2):
#         super(WeightedFocalLoss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         targets = targets.type(torch.long)
#         at = self.alpha.gather(0, targets.data.view(-1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = at*(1-pt)**self.gamma * BCE_loss
#         return F_loss.mean()
