import numpy as np
import pandas as pd
import torch
from gray_zone.models.coral import coral_loss
import numpy as np

from typing import Optional, Union, Sequence # Union added
from torch import Tensor # added for 2nd implementation
import warnings 
import torch.nn as nn
import torch.nn.functional as F
# 8 print commands in total here



#####################################################################################################
# 1 - adapted from kornia's implementation of focal loss

# One-Hot function
def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")

    if not labels.dtype == torch.int64:
        raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)
    # print(f'one_hot output: \n{one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps}') #
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

# Multiclass focal loss function
def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    # alpha: Union[float, torch.Tensor], # SRA 03/16/22 added
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: Optional[float] = None,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.
    Return:
        the computed loss.
    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)
    # print(f'input_soft = \n{input_soft}') #
    log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)
    # print(f'log_input_soft = \n{log_input_soft}') #

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)
    # print(f'weight = \n{weight}') #
    focal = -alpha * weight * log_input_soft
    # print(f'focal = \n{focal}') #
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))
    # print(f'loss_tmp = \n{loss_tmp}') #
    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    # print(f'final loss = \n{loss}') #
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stability. This is no longer
          used.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none', eps: Optional[float] = None) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: Optional[float] = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)

##################################################################################################################
# # 2 - (variable) alpha implementation of Focal Loss
# class FocalLoss(nn.Module):
#     """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
#     It is essentially an enhancement to cross entropy loss and is
#     useful for classification tasks when there is a large class imbalance.
#     x is expected to contain raw, unnormalized scores for each class.
#     y is expected to contain class labels.
#     Shape:
#         - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
#         - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
#     """

#     def __init__(self,
#                  alpha: Optional[Tensor] = None,
#                  gamma: float = 0.,
#                  reduction: str = 'mean',
#                  ignore_index: int = -100):
#         """Constructor.
#         Args:
#             alpha (Tensor, optional): Weights for each class. Defaults to None.
#             gamma (float, optional): A constant, as described in the paper.
#                 Defaults to 0.
#             reduction (str, optional): 'mean', 'sum' or 'none'.
#                 Defaults to 'mean'.
#             ignore_index (int, optional): class label to ignore.
#                 Defaults to -100.
#         """
#         if reduction not in ('mean', 'sum', 'none'):
#             raise ValueError(
#                 'Reduction must be one of: "mean", "sum", "none".')

#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#         self.reduction = reduction

#         self.nll_loss = nn.NLLLoss(
#             weight=alpha, reduction='none', ignore_index=ignore_index)

#     def __repr__(self):
#         arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
#         arg_vals = [self.__dict__[k] for k in arg_keys]
#         arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
#         arg_str = ', '.join(arg_strs)
#         return f'{type(self).__name__}({arg_str})'

#     def forward(self, x: Tensor, y: Tensor) -> Tensor:
#         if x.ndim > 2:
#             # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
#             c = x.shape[1]
#             x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
#             # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
#             y = y.view(-1)

#         unignored_mask = y != self.ignore_index
#         y = y[unignored_mask]
#         if len(y) == 0:
#             return 0.
#         x = x[unignored_mask]

#         # compute weighted cross entropy term: -alpha * log(pt)
#         # (alpha is already part of self.nll_loss)
#         log_p = F.log_softmax(x, dim=-1)
#         ce = self.nll_loss(log_p, y)

#         # get true class column from each row
#         all_rows = torch.arange(len(x))
#         log_pt = log_p[all_rows, y]

#         # compute focal term: (1 - pt)^gamma
#         pt = log_pt.exp()
#         focal_term = (1 - pt)**self.gamma

#         # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
#         loss = focal_term * ce

#         if self.reduction == 'mean':
#             loss = loss.mean()
#         elif self.reduction == 'sum':
#             loss = loss.sum()

#         return loss

##################################################################################################################

class KappaLoss():
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, output, y):
        ohe_y = torch.zeros_like(output)
        batch_size = y.size(0)
        for i in range(batch_size):
            ohe_y[i, y[i].long()] = 1.

        output = torch.nn.Softmax(dim=1)(output)
        W = np.zeros((self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                W[i, j] = abs(i - j) ** 2

        W = torch.from_numpy(W.astype(np.float32)).to(y.device)

        O = torch.matmul(ohe_y.t(), output)
        E = torch.matmul(ohe_y.sum(dim=0).view(-1, 1), output.sum(dim=0).view(1, -1)) / O.sum()

        return (W * O).sum() / ((W * E).sum() + 1e-5)
####################################################################################################################

class FocalKappaLoss():
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, output, y):
        ohe_y = torch.zeros_like(output)
        batch_size = y.size(0)
        for i in range(batch_size):
            ohe_y[i, y[i].long()] = 1.

        output = torch.nn.Softmax(dim=1)(output)
        W = np.zeros((self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                W[i, j] = abs(i - j) ** 2

        W = torch.from_numpy(W.astype(np.float32)).to(y.device)

        O = torch.matmul(ohe_y.t(), output)
        E = torch.matmul(ohe_y.sum(dim=0).view(-1, 1), output.sum(dim=0).view(1, -1)) / O.sum()

        return (W * O).sum() / ((W * E).sum() + 1e-5)

####################################################################################################################

def get_loss(loss_id: str,
             n_class: int,
             is_weighted: bool = False,
             weights: torch.Tensor = None,
             device: str = None):
    """ Get loss function from loss id. Choices between: 'ce', 'mse', 'l1', 'bce', 'mse', 'coral', 'foc' """
    loss = None
    weights = weights.to(device) if is_weighted else None

    if loss_id == 'coral':
        loss = coral_loss
    elif loss_id == 'ce':
        loss = torch.nn.CrossEntropyLoss(weight=weights)
    elif loss_id == 'mse':
        loss = torch.nn.MSELoss()
    elif loss_id == 'qwk':
        loss = KappaLoss(n_classes=n_class)
    elif loss_id == 'l1':
        loss = torch.nn.L1Loss()
    elif loss_id == 'bce':
        loss = torch.nn.BCELoss()
    elif loss_id == 'foc':
        # kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'} # 1 - 91
        # kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'sum'} # 2 - 97 - SRA 03/14/2022 changed 1 to reduction:sum
        # kwargs = {"alpha": 1, "gamma": 2.0, "reduction": 'mean'} # 3 - 98 - SRA 03/14/2022 changed 1 to alpha:1
        # kwargs = {"alpha": weights, "gamma": 2.0, "reduction": 'mean'} # 4 - 99 - SRA 03/14/2022 changed 1 to alpha=weights and used the 2nd (AdeelH) implementation of focal loss; set is_weighted_loss=True
        kwargs = {"alpha": 1, "gamma": 1.5, "reduction": 'mean'} # 3 - 121 - SRA 03/14/2022 changed 1 to gamma:1.5
        kwargs = {"alpha": 1, "gamma": 3, "reduction": 'mean'} # 3 - 122 - SRA 03/14/2022 changed 1 to gamma:1.5
        kwargs = {"alpha": 1, "gamma": 4, "reduction": 'mean'} # 3 - 123 - SRA 03/14/2022 changed 1 to gamma:1.5
        print(f'focal loss params: \n{kwargs}')
        loss = FocalLoss(**kwargs)
    elif loss_id == 'foc_qwk':
        kwargs = {"alpha": weights, "gamma": 2.0, "reduction": 'mean'}
    else:
        raise ValueError("Invalid loss function id. Choices: 'ce', 'mse', 'l1', 'bce', 'qwk', 'coral', 'foc'")

    return loss
