import torch
import torch.nn.functional as F
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        gamma: focusing parameter (>=0)
        alpha: weight per class [alpha0, alpha1], or None
        """
        super().__init__()
        self.gamma    = gamma
        self.alpha    = alpha
        self.reduction= reduction

    def forward(self, logits, targets):
        """
        logits: (B, C)
        targets: (B,) int64
        """
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)  # model confidence for true class
        if self.alpha is not None:
            at = self.alpha.gather(0, targets) 
            ce = at * ce
        loss = ((1-pt)**self.gamma) * ce
        if self.reduction=='mean':
            return loss.mean()
        elif self.reduction=='sum':
            return loss.sum()
        else:
            return loss
