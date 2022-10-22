import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

eps =1e-7
class UACE(nn.Module):

    def __init__(self, alpha: float = 10.0, beta:float=4,gamma:float=1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta 
        self.lamb = lamb

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(input, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        num_classes = int(pred.shape[1])
        label_one_hot = F.one_hot(target, num_classes).float().cuda()
        p0 = pred[torch.arange(pred.shape[0]), target]
        #UWT
        pc = -torch.sum(self.beta*(1-label_one_hot)*(pred*torch.log(pred)), dim=1)+self.gamma*p0
        #NegCE*UWT
        loss = -torch.log(1+self.alpha*p0)*pc
        return loss.mean()