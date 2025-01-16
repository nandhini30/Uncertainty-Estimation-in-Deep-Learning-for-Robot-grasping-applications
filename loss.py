import torch
import torch.nn as nn
import numpy as np

# Define Gaussian Negative Log-Likelihood Loss
criterion = nn.GaussianNLLLoss(reduction='none')

def LaplaceNLLLoss(input, target, scale, eps=1e-06, reduction='mean'):
    """
    Computes the Laplace Negative Log-Likelihood Loss.

    Args:
        input (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.
        scale (torch.Tensor): Scale parameter (variance).
        eps (float): Small value to prevent numerical instability.
        reduction (str): Specifies reduction method ('none', 'mean', 'sum').

    Returns:
        torch.Tensor: Loss value.
    """
    # Validate input shapes
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    scale = scale.view(input.size(0), -1)
    if scale.size(1) != input.size(1) and scale.size(1) != 1:
        raise ValueError("scale is of incorrect size")

    if reduction not in ['none', 'mean', 'sum']:
        raise ValueError(f"{reduction} is not valid")

    if torch.any(scale < 0):
        raise ValueError("scale has negative entry/entries")

    scale = scale.clone()
    with torch.no_grad():
        scale.clamp_(min=eps)

    loss = (torch.log(2 * scale) + torch.abs(input - target) / scale).view(input.size(0), -1).sum(dim=1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def CauchyNLLLoss(input, target, scale, eps=1e-06, reduction='mean'):
    """
    Computes the Cauchy Negative Log-Likelihood Loss.

    Args:
        input (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.
        scale (torch.Tensor): Scale parameter (variance).
        eps (float): Small value to prevent numerical instability.
        reduction (str): Specifies reduction method ('none', 'mean', 'sum').

    Returns:
        torch.Tensor: Loss value.
    """
    # Validate input shapes
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    scale = scale.view(input.size(0), -1)
    if scale.size(1) != input.size(1) and scale.size(1) != 1:
        raise ValueError("scale is of incorrect size")

    if reduction not in ['none', 'mean', 'sum']:
        raise ValueError(f"{reduction} is not valid")

    if torch.any(scale < 0):
        raise ValueError("scale has negative entry/entries")

    scale = scale.clone()
    with torch.no_grad():
        scale.clamp_(min=eps)

    loss = (torch.log(3.14 * scale) + torch.log(1 + ((input - target) ** 2) / scale ** 2)).view(input.size(0), -1).sum(dim=1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

class EvidentialLoss(nn.Module):
    """
    Computes Evidential Loss for uncertainty-aware predictions.

    Args:
        mu (torch.Tensor): Predicted mean.
        alpha (torch.Tensor): Shape parameter.
        beta (torch.Tensor): Scale parameter.
        lamda (torch.Tensor): Precision parameter.
        targets (torch.Tensor): Ground truth values.
    """
    def __init__(self, mu, alpha, beta, lamda, targets):
        super(EvidentialLoss, self).__init__()
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.targets = targets

    def forward(self, mu, alpha, beta, lamda, targets, smooth=1):
        targets = targets.view(-1)
        y = mu.view(-1)
        loga = alpha.view(-1)
        logb = beta.view(-1)
        logl = lamda.view(-1)

        a = torch.exp(loga)
        b = torch.exp(logb)
        l = torch.exp(logl)

        term1 = (torch.exp(torch.lgamma(a - 0.5))) / (4 * torch.exp(torch.lgamma(a)) * l * torch.sqrt(b))
        term2 = 2 * b * (1 + l) + (2 * a - 1) * l * (y - targets) ** 2
        J = term1 * term2

        Kl_divergence = torch.abs(y - targets) * (2 * a + l)
        loss = J + Kl_divergence

        return loss.mean()

def GeneralGaussianNLLLoss(input, target, alpha, beta, eps=1e-06, reduction='none'):
    """
    Computes General Gaussian Negative Log-Likelihood Loss.

    Args:
        input (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.
        alpha (torch.Tensor): Scale parameter (alpha).
        beta (torch.Tensor): Shape parameter (beta).
        eps (float): Small value to prevent numerical instability.
        reduction (str): Specifies reduction method ('none', 'mean', 'sum').

    Returns:
        torch.Tensor: Loss value.
    """
    # Validate input shapes
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    alpha = alpha.view(input.size(0), -1)
    if alpha.size(1) != input.size(1) and alpha.size(1) != 1:
        raise ValueError("alpha is of incorrect size")

    beta = beta.view(input.size(0), -1)
    if beta.size(1) != input.size(1) and beta.size(1) != 1:
        raise ValueError("beta is of incorrect size")

    if reduction not in ['none', 'mean', 'sum']:
        raise ValueError(f"{reduction} is not valid")

    if torch.any(alpha < 0):
        raise ValueError("alpha has negative entry/entries")

    if torch.any(beta < 0):
        raise ValueError("beta has negative entry/entries")

    alpha = alpha.clone()
    beta = beta.clone()
    with torch.no_grad():
        alpha.clamp_(min=eps)
        beta.clamp_(min=eps)

    loss = (torch.abs(input - target) / alpha) ** beta - torch.log(beta) + torch.log(2 * alpha) + torch.lgamma(1 / beta)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss
