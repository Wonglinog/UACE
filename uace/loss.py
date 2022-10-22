import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

eps =1e-7
def f_label(input_tensor,label):
    
    index_tensor = torch.arange(input_tensor.shape[1])

    index_repeat = torch.repeat_interleave(index_tensor.unsqueeze(dim=1),repeats = input_tensor.shape[0], dim=1)
    save_label = torch.zeros(index_repeat.shape[0]-1,index_repeat.shape[1])
    np_save_label = np.array(save_label)
    #print("c",np_save_label)
    #d = np.delete(index_repeat.numpy(),label.unsqueeze(dim=-1).numpy())

    for i in range(len(label)):
        np_save_label[:,i] = np.delete(index_repeat[:,i],label[i])
        #c_label[i,c[:,i]]
    #    print(np_save_label[:,i])
        save_label[:,i] = input_tensor[i,np_save_label[:,i].astype(np.int)]
    #    print(save_label[:,i])
    
    #print("---",torch.transpose(save_label,1,0))
    
    return torch.transpose(save_label,1,0)

class CrossEntropy(nn.Module):
    """Computes the cross-entropy loss

    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self) -> None:
        super().__init__()
        # Use log softmax as it has better numerical properties
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.log_softmax(input)
        
        p = p[torch.arange(p.shape[0]), target]

        loss = -p


        pred = F.softmax(input, dim=1)
        pred = pred[torch.arange(pred.shape[0]), target]
        loss_grad = (1./pred).detach()
        #print("loss_grad",loss_grad)
        return torch.mean(loss), loss_grad

class KCE(nn.Module):


    def __init__(self, alpha: float = 10.0, beta:float=10, gamma:float=1.0,lamb:float=1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta 
        self.lamb = lamb

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred = F.softmax(input, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        num_classes = int(pred.shape[1])
        label_one_hot = F.one_hot(target, num_classes).float().cuda()

        p0 = pred[torch.arange(pred.shape[0]), target]



        pc = self.lamb*torch.sum((1)*pred*(torch.log(pred)), dim=1)+ self.beta*p0# self.beta*(1-torch.sum((1-label_one_hot)*pred, dim=1)) #

        #pc = pc.detach()

        

        loss = -self.gamma*torch.log(1+self.alpha*p0)*pc

        return loss.mean()


class FCrossEntropy(nn.Module):
    """Computes the cross-entropy loss

    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self) -> None:
        super().__init__()
        # Use log softmax as it has better numerical properties
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.log_softmax(input)

        print(p[0,:])
        print(target[0])

        p1 = p[torch.arange(p.shape[0]), target]

        #loss = -p

        f_p = f_label(input.cpu(),target.cpu()).cuda()

        print("f_p",f_p[0,:])
        #f_p = self.softmax(f_p)
        #print(f_p.shape)
        loss_f_p = torch.sum(f_p,dim=-1)

        #print(loss_f_p)
        #print(p1)
        loss = -p1 #- 0.01*loss_f_p
        return torch.mean(loss)





class GeneralizedCrossEntropy(nn.Module):
    """Computes the generalized cross-entropy loss, from `
    "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels"
    <https://arxiv.org/abs/1805.07836>`_

    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`


    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-9
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        # Avoid undefined gradient for p == 0 by adding epsilon
        p += self.epsilon

        loss = (1 - p ** self.q) / self.q


        loss_grad = p**(self.q-1)
        return torch.mean(loss), loss_grad.detach()



class AGCELoss(nn.Module):
    def __init__(self, num_classes=10, a=1, q=2, eps=1e-7, scale=1.):
        super(AGCELoss, self).__init__()
        self.a = a
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = ((self.a+1)**self.q - torch.pow(self.a + torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean() * self.scale

class AUELoss(nn.Module):
    def __init__(self, num_classes=10, a=1.5, q=0.9, eps=eps, scale=1.0):
        super(AUELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.q = q
        self.eps = eps
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (torch.pow(self.a - torch.sum(label_one_hot * pred, dim=1), self.q) - (self.a-1)**self.q)/ self.q
        return loss.mean() * self.scale


class AExpLoss(torch.nn.Module):
    def __init__(self, num_classes=10, a=3, scale=1.0):
        super(AExpLoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = torch.exp(-torch.sum(label_one_hot * pred, dim=1) / self.a)
        return loss.mean() * self.scale


class NCEandMAE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(NCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.mae(pred, labels)


class NCEandAGCE(torch.nn.Module):
    def __init__(self, alpha=1., beta = 1., num_classes=10, a=3, q=1.5):
        super(NCEandAGCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(num_classes=num_classes, scale=alpha)
        self.agce = AGCELoss(num_classes=num_classes, a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.agce(pred, labels)


class NCEandAUE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10, a=6, q=1.5):
        super(NCEandAUE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(num_classes=num_classes, scale=alpha)
        self.aue = AUELoss(num_classes=num_classes, a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.aue(pred, labels)

class NCEandAEL(torch.nn.Module):
    def __init__(self, alpha=1., beta=4., num_classes=10, a=2.5):
        super(NCEandAEL, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(num_classes=num_classes, scale=alpha)
        self.aue = AExpLoss(num_classes=num_classes, a=a, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.aue(pred, labels)




    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.agce(pred, labels)


class Unhinged(nn.Module):
    """Computes the Unhinged (linear) loss, from
    `"Learning with Symmetric Label Noise: The Importance of Being Unhinged"

    <https://arxiv.org/abs/1505.07634>`_


    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = 1 - p

        return torch.mean(loss)


class PHuberCrossEntropy(nn.Module):
    """Computes the partially Huberised (PHuber) cross-entropy loss, from
    `"Can gradient clipping mitigate label noise?"
    <https://openreview.net/pdf?id=rklB76EKPr>`_

    Args:
        tau: clipping threshold, must be > 1


    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, tau: float = 10) -> None:
        super().__init__()
        self.tau = tau

        # Probability threshold for the clipping
        self.prob_thresh = 1 / self.tau
        # Negative of the Fenchel conjugate of base loss at tau
        self.boundary_term = math.log(self.tau) + 1

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = torch.empty_like(p)

        loss_grad = torch.empty_like(p)

        clip = p <= self.prob_thresh
        loss[clip] = -self.tau * p[clip] + self.boundary_term
        loss[~clip] = -torch.log(p[~clip])

        p_grad = torch.ones_like(p)
        loss_grad[clip] = self.tau * p_grad[clip] #+ self.boundary_term
        loss_grad[~clip] = 1./p[~clip]



        return torch.mean(loss),loss_grad.detach()


class PHuberGeneralizedCrossEntropy(nn.Module):
    """Computes the partially Huberised (PHuber) generalized cross-entropy loss, from
    `"Can gradient clipping mitigate label noise?"
    <https://openreview.net/pdf?id=rklB76EKPr>`_

    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`
        tau: clipping threshold, must be > 1


    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, q: float = 0.7, tau: float = 10) -> None:
        super().__init__()
        self.q = q
        self.tau = tau

        # Probability threshold for the clipping
        self.prob_thresh = tau ** (1 / (q - 1))
        # Negative of the Fenchel conjugate of base loss at tau
        self.boundary_term = tau * self.prob_thresh + (1 - self.prob_thresh ** q) / q

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = torch.empty_like(p)
        clip = p <= self.prob_thresh
        loss[clip] = -self.tau * p[clip] + self.boundary_term
        loss[~clip] = (1 - p[~clip] ** self.q) / self.q

        return torch.mean(loss)




class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        
        label_one_hot = F.one_hot(labels, int(pred.shape[1])).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        #print("+++++++++++++",rce)
        return self.scale * rce.mean()



class PoCELoss(nn.Module):

    def __init__(self, alpha: float = 10.0, gamma:float=1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred = F.softmax(input, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        num_classes = int(pred.shape[1]) 
        label_one_hot = F.one_hot(target, num_classes).float().cuda()

        p0 = pred[torch.arange(pred.shape[0]), target]
        #pc = -torch.sum((1-label_one_hot)*torch.log(0.2/(num_classes-1)+0.8*pred)*(0.2/(num_classes-1)+0.8*pred), dim=1)
        #pc = -torch.sum((1-label_one_hot)*torch.log(pred)*(pred), dim=1)
        #pc = -torch.sum((1-label_one_hot)*torch.log(1-0.7*pred)*(1-0.7*pred), dim=1)
        #pc = torch.sum(torch.log(1+self.alpha*pred*pred),dim=1)

        pc = -torch.sum((1-label_one_hot)*(pred*torch.log(pred)), dim=1)+self.alpha*p0

        loss = (self.gamma*torch.log(1+self.alpha*p0)*pc).detach() 

        return loss.mean()


class UACE(nn.Module):

    def __init__(self, alpha: float = 10.0, beta:float=10, gamma:float=1.0,lamb:float=1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta 
        self.lamb = lamb
        self.nce = NormalizedCrossEntropy(num_classes=10, scale=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred = F.softmax(input, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        num_classes = int(pred.shape[1])
        label_one_hot = F.one_hot(target, num_classes).float().cuda()

        p0 = pred[torch.arange(pred.shape[0]), target]

        pc = -torch.sum(self.lamb*(1-label_one_hot)*(pred*torch.log(pred)), dim=1)+self.beta*(p0)



        loss = -self.gamma*torch.log(1+self.alpha*p0)*(pc)

        #print(loss)
        
        loss_grad = self.alpha/(1+self.alpha*p0)*pc

        return loss.mean(), loss_grad.detach()

class NegCE(nn.Module):

    def __init__(self, alpha: float = 10.0, beta:float=10, gamma:float=1.0,lamb:float=1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta 
        self.lamb = lamb
        self.nce = NormalizedCrossEntropy(num_classes=10, scale=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred = F.softmax(input, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        num_classes = int(pred.shape[1])
        label_one_hot = F.one_hot(target, num_classes).float().cuda()

        p0 = pred[torch.arange(pred.shape[0]), target]

        pc = -torch.sum(self.lamb*(1-label_one_hot)*(pred*torch.log(pred)), dim=1)+self.beta*(p0)



        loss = -self.gamma*torch.log(1+self.alpha*p0)#*(pc)

        #print(loss)
        
        loss_grad = self.alpha/(1+self.alpha*p0)#*pc

        return loss.mean(), loss_grad.detach()


class NNeCELoss(nn.Module):


    def __init__(self, beta: float = 10.0, lamb:float=1.0) -> None:
        super().__init__()
        self.beta = beta
        self.lamb = lamb

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred = F.softmax(input, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(target, int(pred.shape[1])).float().cuda()

        p0 = pred[torch.arange(pred.shape[0]), target]
        pc = -torch.sum((1-label_one_hot)*(torch.log(pred)*(pred)), dim=1)+p0

        loss = self.lamb*torch.log(1-self.beta*p0)*pc

        return loss.mean()


class NeCELoss(nn.Module):

    def __init__(self, beta: float = 10.0, lamb:float=1.0) -> None:
        super().__init__()
        self.beta = beta
        self.lamb = lamb

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred = F.softmax(input, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(target, int(pred.shape[1])).float().cuda()

        p0 = pred[torch.arange(pred.shape[0]), target]
        
        loss = self.lamb*torch.log(1-self.beta*p0)

        return loss.mean()

class PoCEandNeCE(nn.Module):

    def __init__(self, alpha: float = 10.0, beta: float = 0.4,lamb:float=1.0,gamma:float=1.0) -> None:
        super().__init__()

        self.PoCELoss = PoCELoss(alpha = alpha, gamma = gamma)
        self.NeCELoss = NeCELoss(beta = beta, lamb = lamb)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return self.PoCELoss(input,target) + self.NeCELoss(input,target)


class NNeCEandNeCE(nn.Module):

    def __init__(self, alpha: float = 10.0, beta: float = 0.4,lamb:float=1.0,gamma:float=1.0) -> None:
        super().__init__()

        self.NNeCELoss = NNeCELoss(beta = alpha, lamb = gamma)
        self.NeCELoss = NeCELoss(beta = beta, lamb = lamb)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return self.NNeCELoss(input,target) + self.NeCELoss(input,target)

class NNeCEandRCE(nn.Module):

    def __init__(self, alpha: float = 10.0, beta: float = 0.4,lamb:float=1.0,gamma:float=1.0,num_classes:int=10) -> None:
        super().__init__()

        self.NNeCELoss = NNeCELoss(beta = alpha, lamb = gamma)
        self.RCELoss = ReverseCrossEntropy(scale = lamb,num_classes = num_classes)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return self.NNeCELoss(input,target) + self.RCELoss(input,target)

class PoCEandRCE(nn.Module):

    def __init__(self, alpha: float = 10.0, beta: float = 0.4, gamma:float=1.0,num_classes:int=10) -> None:
        super().__init__()

        self.PoCELoss = PoCELoss(alpha = alpha, gamma = gamma)
        self.RCE = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return self.PoCELoss(input,target) + self.RCE(input,target)

class NCEandNeCE(nn.Module):

    def __init__(self, num_classes:int=10, beta: float = 0.4,lamb:float=1.0,scale:float=1.0) -> None:
        super().__init__()

        self.NCE = NormalizedCrossEntropy(scale=1.0, num_classes=num_classes)
        self.NeCELoss = NeCELoss(beta = beta, lamb = lamb)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return self.NCE(input,target) + self.NeCELoss(input,target)

class NCEandPPoCE(nn.Module):

    def __init__(self, num_classes:int=10, alpha: float = 10.0, beta:float=4, gamma:float=1.0,lamb:float=1.0) -> None:
        super().__init__()

        self.NCE = NormalizedCrossEntropy(scale=1.0, num_classes=num_classes)
        self.PPoCELoss = PPoCELoss(alpha = alpha, beta=beta, gamma = gamma, lamb=lamb)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return self.NCE(input,target) + self.PPoCELoss(input,target)


class PoGCEandNeCE(nn.Module):

    def __init__(self, num_classes:int=10, beta: float = 0.4,lamb:float=1.0,q:float=0.7) -> None:
        super().__init__()

        self.PoGCE = PoGCE(q=q)
        self.NeCELoss = NeCELoss(beta = beta, lamb = lamb)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return self.PoGCE(input,target) + self.NeCELoss(input,target)


class NPCE(nn.Module):

    def __init__(self, num_classes:int=10,alpha: float = 10.0, beta:float=10, gamma:float=1.0,lamb:float=1.0) -> None:
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta 
        self.lamb = lamb

        self.NCE = NormalizedCrossEntropy(scale=self.lamb, num_classes=num_classes)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred = F.softmax(input, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        num_classes = int(pred.shape[1])
        label_one_hot = F.one_hot(target, num_classes).float().cuda()

        p0 = pred[torch.arange(pred.shape[0]), target]


        pce = (-self.gamma*torch.log(1+self.alpha*p0)*self.beta*p0).mean()
        nce = self.NCE(input,target)-1

        return pce+nce



class PoGCE(nn.Module):

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-9
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)        
        p = torch.clamp(p, min=eps, max=1.0)

        label_one_hot = F.one_hot(target, int(p.shape[1])).float().cuda()

        pc = -torch.sum((1-label_one_hot)*torch.log(p)*p, dim=1)

        p = p[torch.arange(p.shape[0]), target]


        loss = (pc)*((1 - p ** self.q) / self.q-(1/0.7))

        return torch.mean(loss)


class BCCE(nn.Module):

    def __init__(self, alpha: float = 10.0, beta: float = 0.4,lamb:float=1.0,gamma:float=1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb
        self.gamma = gamma
        #self.epsilon = 1e-7
        self.rce = ReverseCrossEntropy(scale=1, num_classes=10)
        self.nce  = NormalizedCrossEntropy(scale=1.0, num_classes=10)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        rce = self.rce(input,target)
        
        nce = self.nce(input, target)


        pred = F.softmax(input, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(target, int(pred.shape[1])).float().cuda()
 



        p0 = pred[torch.arange(pred.shape[0]), target]
        # #p1 = p[torch.arange(p.shape[0]), target1]
        loss = torch.empty_like(p0)
        clip = p0 <= 1e-4


        # 
        ce = -torch.log(p0)



        pred_p = -(torch.log(1-pred)).sum(dim=1).detach()#
        #print("pred",pred_p)

        sum_p = torch.repeat_interleave(pred_p.unsqueeze(dim=1), repeats=10, dim=1)


        pc = -torch.sum((1-label_one_hot)*torch.log(pred)*pred, dim=1)



        alpha = self.alpha

        pce = -torch.log(1+alpha*p0)*pc

        

        beta = self.beta#*((1-sum_p))
        tce =  torch.log(1-beta*p0)


        ce = -torch.log(p0)

        loss[clip]  = self.gamma*pce[clip]  + self.lamb*tce[clip]   #+ 0.05*ce[clip]
        loss[~clip] = self.gamma*pce[~clip] + self.lamb*tce[~clip] # + 0.1*ce[clip]


        return loss.mean()




class NCCE(nn.Module):

    def __init__(self, alpha: float = 10.0, beta: float = 0.4,lamb:float=5.0,gamma:float=2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb
        self.gamma = gamma
        self.epsilon = 1e-7
        self.softmax = nn.Softmax(dim=1)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=10)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self.softmax(input)
        pred = torch.clamp(pred, self.epsilon, 1.0)
        rce  = self.rce(input,target)

        label_one_hot = F.one_hot(target, pred.shape[1]).float()
        #label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        #print(label_one_hot.shape)
        pce = -torch.sum(label_one_hot*torch.log(1+self.alpha*pred),dim=-1)

        #nce = torch.sum((1-label_one_hot)*torch.log(1-self.beta*pred),dim=-1)

        #nce = torch.sum(pred*torch.log(1+self.alpha*pred),dim=-1)

        nce = -torch.sum(label_one_hot*torch.log(1-self.beta*pred),dim=-1)

        loss = self.gamma*pce+self.lamb*nce#+0.2*rce



        return torch.mean(loss)




# class SCE(nn.Module):

#     def __init__(self, alpha: float = 10.0, beta: float = 0.4) -> None:
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.softmax = nn.Softmax(dim=1)
#         self.cross_entropy = torch.nn.CrossEntropyLoss()

#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:


#         p = self.log_softmax(input)
#         p = p[torch.arange(p.shape[0]), target]
#         #loss = -p
#         ce = torch.mean(-p)

#         #ce = self.cross_entropy(input, target)
#         pred = self.softmax(input)
        
#         # RCE
#         pred = torch.clamp(pred, min=1e-7, max=1.0)
#         label_one_hot = F.one_hot(target, pred.shape[1]).float().cuda()
#         label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
#         rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

#         #print("rce",label_one_hot)

#         # Loss
#         loss = self.alpha * ce + self.beta * rce.mean()

#         return loss



class SCE(torch.nn.Module):
    def __init__(self, alpha: float = 10.0, beta: float = 0.4) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=0, max=1.0)
        label_one_hot = F.one_hot(labels, pred.shape[1]).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()

        loss_grad = self.alpha*(1./pred) + self.beta*(1)

        print(loss,loss_grad.detach())
        return loss,loss_grad.detach()



class TCE(nn.Module):

    def __init__(self, beta: float = 2) -> None:
        super().__init__()
        self.beta = beta
        self.epsilon = 0
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self.softmax(input)

        pred = torch.clamp(pred, self.epsilon, 1.0-self.epsilon)


        pred = pred[torch.arange(pred.shape[0]), target]

        loss = torch.zeros_like(pred)#.unsqueeze(0).expand(int(self.beta),-1)

        loss_grad = torch.zeros_like(pred)

        # print("loss-0",loss)
        # #print(loss.shape)
        # for j in range(int(self.beta)):
        #     q = int(j+1)
        #     loss[j,:] = ((1-pred)**q)/q
        # print("loss-1",loss)
        # 
        for j in range(int(self.beta)):
            q = int(j+1)
            loss += ((1-pred)**q)/q
            loss_grad += (1-pred)**(q-1)

        return loss.mean(), loss_grad.detach()


class MeanAbsoluteError(nn.Module):
    """Computes the cross-entropy loss

    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self,scale: float = 2) -> None:
        super().__init__()
        # Use log softmax as it has better numerical properties
        self.softmax = nn.Softmax(dim=1)
        self.scale = scale

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred = self.softmax(input)
        label_one_hot = F.one_hot(target, pred.shape[1]).float()
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #

        return self.scale * mae.mean()


#-------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------#

class NLNL(torch.nn.Module):
    def __init__(self, train_loader, num_classes, ln_neg=1):
        super(NLNL, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.
        if not hasattr(train_loader.dataset, 'targets'):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (torch.from_numpy(np.array(train_loader.dataset.targets)) == i).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        labels_neg = (labels.unsqueeze(-1).repeat(1, self.ln_neg)
                      + torch.LongTensor(len(labels), self.ln_neg).to(self.device).random_(1, self.num_classes)) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes-1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(labels)*self.ln_neg

        s_neg = torch.log(torch.clamp(1. - F.softmax(pred, 1), min=1e-5, max=1.))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size()).to(self.device)
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
        loss = ((loss+loss_neg) / (float((labels >= 0).sum())+float((labels_neg[:, 0] >= 0).sum())))
        return loss




class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, int(pred.shape[1])).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        pred0 = pred[torch.arange(pred.shape[0]), labels]

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        loss_grad = self.alpha*(1./pred0) + self.beta*(1)

        #print(loss, loss_grad.detach())
        return loss, loss_grad.detach()



class NormalizedReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedReverseCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, int(pred.shape[1])).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        normalizor = 1 / 4 * (int(pred.shape[1]) - 1)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * normalizor * rce.mean()


class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, int(pred.shape[1])).float().cuda()
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()



class NCELoss(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * loss.mean()


class GCE(torch.nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GCE, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, int(pred.shape[1])).float().to(self.device)
        gce = 2*(1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()



class NormalizedGeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0, q=0.7):
        super(NormalizedGeneralizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, int(pred.shape[1])).float().to(self.device)
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = int(pred.shape[1]) - pred.pow(self.q).sum(dim=1)
        ngce = numerators / denominators
        return self.scale * ngce.mean()


class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(MeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels,int(pred.shape[1])).float().cuda()
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        return self.scale * mae.mean()


class NormalizedMeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedMeanAbsoluteError, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, int(pred.shape[1])).float().to(self.device)
        normalizor = 1 / (2 * (int(pred.shape[1]) - 1))
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * normalizor * mae.mean()



class NCEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)


class NCEandMAE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(NCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.mae(pred, labels)



class GCEandMAE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.gce = GCE(num_classes=num_classes, q=q)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.mae(pred, labels)



class GCEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.gce = GCE(num_classes=num_classes, q=q)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.rce(pred, labels)


class GCEandNCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandNCE, self).__init__()
        self.num_classes = num_classes
        self.gce = GCE(num_classes=num_classes, q=q)
        self.nce = NormalizedCrossEntropy(num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.nce(pred, labels)



class NGCEandNCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandNCE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(scale=alpha, q=q, num_classes=num_classes)
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.nce(pred, labels)



class NGCEandMAE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(scale=alpha, q=q, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.mae(pred, labels)



class NGCEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(scale=alpha, q=q, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.rce(pred, labels)



class MAEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(MAEandRCE, self).__init__()
        self.num_classes = num_classes
        self.mae = MeanAbsoluteError(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.mae(pred, labels) + self.rce(pred, labels)



class FocalLoss(torch.nn.Module):
    '''
        https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    '''

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()



class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, scale=1.0, gamma=0, num_classes=10, alpha=None, size_average=True):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()



class NFLandNCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandNCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(scale=alpha, gamma=gamma, num_classes=num_classes)
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.nce(pred, labels)



class NFLandMAE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandMAE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(scale=alpha, gamma=gamma, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.mae(pred, labels)


class NFLandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(scale=alpha, gamma=gamma, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.rce(pred, labels)



class DMILoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(DMILoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        outputs = F.softmax(output, dim=1)
        targets = target.reshape(target.size(0), 1).cpu()
        y_onehot = torch.FloatTensor(target.size(0), self.num_classes).zero_()
        y_onehot.scatter_(1, targets, 1)
        y_onehot = y_onehot.transpose(0, 1).cuda()
        mat = y_onehot @ outputs
        return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)



class NNCELoss(nn.Module):
    def __init__(self,num_classes):
        super(NNCELoss, self).__init__()
        self.eps = 1e-7

    def forward(self, pred, labels):

        #print(pred)
        
        

        #print(argmax_pred)

        target_neg = (labels.unsqueeze(-1).cpu()+ torch.LongTensor(len(labels), 1).random_(1, int(pred.shape[1]))) % int(pred.shape[1])
        target_neg = target_neg.squeeze(-1).cuda()


        
        norms = torch.norm(pred, p=2, dim=-1, keepdim=True) + self.eps


        # symmetric noise-0.2- alpha=5; noise-0.4 -alpha=2.5
        logit_norm = torch.div(pred, norms) /2#5#1.2



        snorm = F.softmax(logit_norm, dim=1)+self.eps#+0.5#/3# np.random.uniform(0.5,1)#0.5#self.eps

        #snorm = snorm**0.5#torch.clamp(snorm, min=1e-7, max=0.5)

        spred = F.softmax(pred, dim=1)+self.eps 

        #p0 = snorm[torch.arange(snorm.shape[0]), labels]
        argmax_snorm = torch.max(snorm,axis=1)
        # #print(argmax_snorm[1])
        for k in range(len(target_neg)):
            #if argmax_snorm[1][k] != labels[k]:# and argmax_snorm[0][k]>0.5:max(argmax_snorm[0][k],0.6)
            #if np.random.rand() >=  0.6 and argmax_snorm[1][k] != labels[k]:
            #
            #pk = np.random.rand()+min(argmax_snorm[0][k]-0.7,0)  max(argmax_snorm[0][k],0.8) 
            if np.random.rand() <= argmax_snorm[0][k] and argmax_snorm[1][k] != labels[k]:
                target_neg[k] = argmax_snorm[1][k]


                #print("-1-")
            #else:
                #print("-0-")
            # elif argmax_snorm[1][k] == labels[k] and argmax_snorm[0][k]>0.5:
            #     target_neg[k] = labels[k]


        


        label_one_hot = F.one_hot(labels, int(logit_norm.shape[1])).float().cuda()
        label_neg_one_hot = F.one_hot(target_neg, int(logit_norm.shape[1])).float().cuda()


        loss1 = (((1-label_one_hot)*snorm.log()).sum(dim=1)).mean()-((label_neg_one_hot)*snorm.log()).sum(dim=1).mean()#
        #loss2 = ((label_one_hot)*snorm.log()).sum(dim=1).mean()+((label_one_hot)*(1-snorm).log()).sum(dim=1).mean()
        #loss3 = F.cross_entropy(spred, labels)
        #print(loss)

        #print(loss, norms.squeeze().detach())
        loss = 10*loss1#loss1+0.5*loss2
        return loss, norms.squeeze().detach()# norms.squeeze() 

class Nlplus(torch.nn.Module):
    def __init__(self, num_classes):
        super(DeCCEandMAE, self).__init__()
        self.device = device
        self.num_classes = num_classes

    def forward(self, output, target):
        pred = F.softmax(output, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(target, int(pred.shape[1])).float().to(self.device)
        #print("target",target)

        target_neg = (target.unsqueeze(-1).cpu()+ torch.LongTensor(len(target), 1).random_(1, int(pred.shape[1]))) % int(pred.shape[1])
        target_neg = target_neg.squeeze(-1).to(self.device)

        #print("target_neg",target_neg)

        label_neg_one_hot = F.one_hot(target_neg, int(pred.shape[1])).float().to(self.device)

        assert target_neg.max() <= int(pred.shape[1])-1 
        assert target_neg.min() >= 0
        #print((target_neg.unsqueeze(-1) != target.unsqueeze(-1)).sum())
        assert (target_neg.unsqueeze(-1) != target.unsqueeze(-1)).sum() == len(target)

        manual_grad = torch.zeros_like(pred)


        weight_nnl_y = torch.zeros([len(target_neg),1],dtype=torch.float).cuda()
        weight_nnl_k = torch.zeros([len(target_neg),1],dtype=torch.float).cuda()
        
        for k in range(len(target)):
            weight_nnl_k[k] = pred[k][target_neg[k]] 
            weight_nnl_y[k] = pred[k][target[k]]

            if weight_nnl_k[k] >= weight_nnl_y[k]:
                t = 1-(weight_nnl_k[k]-weight_nnl_y[k])
            else:
                t = 1-(weight_nnl_k[k]-weight_nnl_y[k])

            #manual_grad_2[k] = -(weight_nnl[k]*original_pred[k])*t
            manual_grad[k][target_neg[k]] = -(weight_nnl_k[k]*(weight_nnl_y[k]+weight_nnl_k[k]))*t-weight_nnl_k[k]*(1-weight_nnl_k[k])*t
            manual_grad[k][target[k]] = (weight_nnl_k[k])*t+(weight_nnl_k[k]*weight_nnl_y[k])*t

        loss = -torch.mean(torch.sum(manual_grad.detach()*output,dim=-1))
        return loss 