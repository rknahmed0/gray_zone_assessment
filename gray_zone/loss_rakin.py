import numpy as np
import pandas as pd
import torch
from gray_zone.models.coral import coral_loss
from typing import Optional, Union, Sequence 
from torch import Tensor
import warnings 
import torch.nn as nn
import torch.nn.functional as F
# throoughly tested for in notebook from 03/19
##################################################################################################
# 1 - adapted from kornia's implementation of focal loss with variable alpha based on weights

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
    print(f'one_hot output: \n{one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps}') #
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
#     return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) # removed eps 03/17


# Multiclass focal loss function
def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: Union[float, torch.Tensor] = 1.0,
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
    print(f'input_soft = \n{input_soft}') #
    log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)
    print(f'log_input_soft = \n{log_input_soft}') #

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)
    print(f'weight = torch.pow(-input_soft + 1.0, gamma)\n{weight}') #
    focal = -1 * weight * log_input_soft
    if not(isinstance(alpha, torch.Tensor)):
        if len([alpha]) == 1:
            alpha = np.array([alpha] * input.shape[1])
            alpha = torch.from_numpy(alpha.astype(np.float32)).to(target.device)        
    adjusted_alpha = torch.matmul(alpha, target_one_hot.t())
    print(f'adjusted_alpha = \n{adjusted_alpha}')
    print(f'focal = \n{focal}') #
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))
    print(f'loss_tmp = \n{loss_tmp}')
    loss_tmp = adjusted_alpha * loss_tmp
    print(f'loss_tmp = \n{loss_tmp}')
    # print(f'loss_tmp = \n{loss_tmp}') #
    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    print(f'final loss = \n{loss}') #
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

    def __init__(self, alpha: Union[float, torch.Tensor] = 1.0, gamma: float = 2.0, reduction: str = 'none', eps: Optional[float] = None) -> None:
        super().__init__()
        self.alpha: Union[float, torch.Tensor] = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: Optional[float] = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)

##################################################################################################################
# # 2 - AdeelH implementation of Focal Loss allowing for variable alpha based on weights

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
# 3. quadratic weighted kappa loss implementation based on sum(W*O)/sum(W*E)

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

#####################################################################################################################################
# 4. focal kappa loss implementation, allowing for (1-pt)^y adjustment to either num or num_denom (verified correct on 03/19 notebook)

class FocalKappaLoss():
    def __init__(self, n_classes, alpha: Union[float, list, torch.Tensor] = 1.0, gamma: float = 2.0, foc_adjustment: str = 'num'):
        self.n_classes = n_classes
        self.alpha: Union[float, list, torch.Tensor] = alpha
        self.gamma: float = gamma
        self.foc_adjustment: str = foc_adjustment

    def __call__(self, output, y):
        ohe_y = torch.zeros_like(output)
        batch_size = y.size(0)
        for i in range(batch_size):
            ohe_y[i, y[i].long()] = 1.
        output = torch.nn.Softmax(dim=1)(output)
        # Calculate W
        W = np.zeros((self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                W[i, j] = abs(i - j) ** 2
        W = torch.from_numpy(W.astype(np.float32)).to(y.device)
        # Calculate O and O_foc_adj
        O = torch.matmul(ohe_y.t(), output)
        # Focal adjusted to O to get O_foc_adj for final output
        focal_weight = torch.pow(-output + 1.0, self.gamma) # output is already softmax
        if not(isinstance(self.alpha, torch.Tensor)):
            if isinstance(self.alpha, float):
                self.alpha = np.array([self.alpha] * self.n_classes)
                self.alpha = torch.from_numpy(self.alpha.astype(np.float32)).to(y.device)
            elif isinstance(self.alpha, list):
                if len(self.alpha) == self.n_classes:
                    self.alpha = np.array(self.alpha)
                    self.alpha = torch.from_numpy(self.alpha.astype(np.float32)).to(y.device)
            else: print('Error: alpha must be a float, 1 x n_classes list or 1 x n_classes torch.Tensor')
        adjusted_alpha = torch.matmul(self.alpha, ohe_y.t()).view(-1,1)        
        F_gt = torch.einsum('bc...,bc...->b...', (ohe_y, focal_weight)).view(-1,1)
        F_gt_alpha = adjusted_alpha * F_gt
        output_foc_adj = F_gt_alpha * output
        O_foc_adj = torch.matmul(ohe_y.t(), output_foc_adj)
        # Calculate E and foc_num_final_loss
        E = torch.matmul(ohe_y.sum(dim=0).view(-1, 1), output.sum(dim=0).view(1, -1)) / O.sum()
        foc_num_final_loss = (W * O_foc_adj).sum() / ((W * E).sum() + 1e-5)
        # Calculate E_foc_adj and foc_num_denom_final_loss
        E_foc_adj = torch.matmul(ohe_y.sum(dim=0).view(-1, 1), output_foc_adj.sum(dim=0).view(1, -1)) / O.sum()
        foc_num_denom_final_loss = (W * O_foc_adj).sum() / ((W * E_foc_adj).sum() + 1e-5)
        # return final loss based on 'num' or 'num_denom' foc_adjustment to kappa
        if self.foc_adjustment == 'num':
            final_loss = foc_num_final_loss
        elif self.foc_adjustment == 'num_denom':
            final_loss = foc_num_denom_final_loss
        #
        return final_loss

#############################################################################################################################
# 4. focal kappa loss LC, allowing for kappa_coeff adjustment to either num or num_denom (verified correct on 03/19 notebook)

class FocalKappaLossLC():
    def __init__(self, n_classes, alpha: Union[float, list, torch.Tensor] = 1.0, gamma: float = 2.0, reduction: str = 'none', foc_coeff: list = [0, 0, 0], kappa_coeff: list = [1, 1, 1], kappa_adjustment: str = 'num'):
        self.n_classes = n_classes
        self.alpha: Union[float, list, torch.Tensor] = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.foc_coeff: list = foc_coeff
        self.kappa_coeff: list = kappa_coeff
        self.kappa_adjustment: str = kappa_adjustment
            
    def __call__(self, output, y):
        ## Kappa loss computation
        ohe_y = torch.zeros_like(output)
        batch_size = y.size(0)
        for i in range(batch_size):
            ohe_y[i, y[i].long()] = 1.
        output_soft = torch.nn.Softmax(dim=1)(output)
        # Calculate W
        W = np.zeros((self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                W[i, j] = abs(i - j) ** 2
        W = torch.from_numpy(W.astype(np.float32)).to(y.device)
        # Calculate O and O_kappa_coeff_adj
        O = torch.matmul(ohe_y.t(), output_soft)
        # adjustment for kappa_coeff 
        if not(isinstance(self.kappa_coeff, torch.Tensor)):
            if isinstance(self.kappa_coeff, list):
                if len(self.kappa_coeff) == self.n_classes:
                    self.kappa_coeff = np.array(self.kappa_coeff)
                    self.kappa_coeff = torch.from_numpy(self.kappa_coeff.astype(np.float32)).to(y.device)  
            else: print('Error: kappa_coeff must be a 1 x n_classes list or 1 x n_classes torch.Tensor')
        adjusted_kappa_coeff = torch.matmul(self.kappa_coeff, ohe_y.t()).view(-1,1)        
        output_kappa_coeff_adj = adjusted_kappa_coeff * output_soft
        O_kappa_coeff_adj = torch.matmul(ohe_y.t(), output_kappa_coeff_adj)
        # Calculate E and kappa_coeff_num_final_loss
        E = torch.matmul(ohe_y.sum(dim=0).view(-1, 1), output_soft.sum(dim=0).view(1, -1)) / O.sum()
        kappa_coeff_num_final_loss = (W * O_kappa_coeff_adj).sum() / ((W * E).sum() + 1e-5)
        # Calculate E_kappa_coeff_adj and kappa_coeff_num_denom_final_loss
        E_kappa_coeff_adj = torch.matmul(ohe_y.sum(dim=0).view(-1, 1), output_kappa_coeff_adj.sum(dim=0).view(1, -1)) / O.sum()
        kappa_coeff_num_denom_final_loss = (W * O_kappa_coeff_adj).sum() / ((W * E_kappa_coeff_adj).sum() + 1e-5)
        # return final loss based on 'num' or 'num_denom' foc_adjustment
        if self.kappa_adjustment == 'num':
            final_kappa_loss = kappa_coeff_num_final_loss
        elif self.kappa_adjustment == 'num_denom':
            final_kappa_loss = kappa_coeff_num_denom_final_loss
        
        ## Focal loss computation
        focal_weight = torch.pow(-output_soft + 1.0, self.gamma) # output is already softmax
        # adjustment for alpha
        if not(isinstance(self.alpha, torch.Tensor)):
            if isinstance(self.alpha, float):
                self.alpha = np.array([self.alpha] * self.n_classes)
                self.alpha = torch.from_numpy(self.alpha.astype(np.float32)).to(y.device)
            elif isinstance(self.alpha, list):
                if len(self.alpha) == self.n_classes:
                    self.alpha = np.array(self.alpha)
                    self.alpha = torch.from_numpy(self.alpha.astype(np.float32)).to(y.device)
            else: print('Error: alpha must be a float, 1 x n_classes list or 1 x n_classes torch.Tensor')
        adjusted_alpha = torch.matmul(self.alpha, ohe_y.t()).view(-1,1)        
        # adjustment for foc_coeff
        if not(isinstance(self.foc_coeff, torch.Tensor)):
            if isinstance(self.foc_coeff, list):
                if len(self.foc_coeff) == self.n_classes:
                    self.foc_coeff = np.array(self.foc_coeff)
                    self.foc_coeff = torch.from_numpy(self.foc_coeff.astype(np.float32)).to(y.device)  
            else: print('Error: focal_coeff must be a 1 x n_classes list or 1 x n_classes torch.Tensor')
        adjusted_foc_coeff = torch.matmul(self.foc_coeff, ohe_y.t()).view(-1,1)        
        # loss computation steps
        log_output_soft = torch.nn.LogSoftmax(dim=1)(output)
        focal_loss = -1 * focal_weight * log_output_soft # output is already softmax
        F_gt = torch.einsum('bc...,bc...->b...', (ohe_y, focal_loss)).view(-1,1)
        F_gt_alpha = adjusted_alpha * F_gt
        F_gt_alpha_foc_coeff = adjusted_foc_coeff * F_gt_alpha
        if self.reduction == 'none':
            final_focal_loss = F_gt_alpha_foc_coeff
        elif self.reduction == 'mean':
            final_focal_loss = torch.mean(F_gt_alpha_foc_coeff)
        elif self.reduction == 'sum':
            final_focal_loss = torch.sum(F_gt_alpha_foc_coeff)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {reduction}")
        # Final loss which is a linear combination of focal_loss and kappa_loss with coefficients by ground truth individually adjusted within each
        final_loss = final_focal_loss + final_kappa_loss
        return final_loss

#######################################################################################################################################################
# get_loss function, used in run_model.py

def get_loss(loss_id: str,
             n_class: int,
             gamma: float = 2.0,
             foc_kappa_adjustment: str = 'num',
             foc_coeff: list = None,
             kappa_coeff: list = None,
             is_weighted: bool = False,
             weights: torch.Tensor = None,
             device: str = None):
    """ Get loss function from loss id. Choices between: 'ce', 'mse', 'l1', 'bce', 'mse', 'coral', 'foc', 'foc_qwk', 'foc_qwk_LC' """
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
        # kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'} # 1 - 91, 92, 93, 94
        # kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'sum'} # 2 - 97 - SRA 03/14/2022 changed 1 to reduction:sum
        # kwargs = {"alpha": 1, "gamma": 2.0, "reduction": 'mean'} # 3 - 98 - SRA 03/14/2022 changed 1 to alpha:1
        # kwargs = {"alpha": weights, "gamma": 2.0, "reduction": 'mean'} # 4 - 99 - SRA 03/14/2022 changed 1 to alpha=weights and used the 2nd (AdeelH) implementation of focal loss; set is_weighted_loss=True
        # kwargs = {"alpha": 1, "gamma": 1.5, "reduction": 'mean'} # 3 - 98b - SRA 03/14/2022 changed 1 to gamma:1.5
        # kwargs = {"alpha": 1, "gamma": 3, "reduction": 'mean'} # 3 - 98c - SRA 03/14/2022 changed 1 to gamma:3
        # kwargs = {"alpha": 1, "gamma": 4, "reduction": 'mean'} # 3 - 98d - SRA 03/14/2022 changed 1 to gamma:4
        ## kwargs skeleton
        # kwargs = {"alpha": 1.0, "gamma": gamma, "reduction": 'mean'} # specific to original Kornia implementation (1a in 3/19 notebook)
        kwargs = {"alpha": 1.0 if weights==None else weights, "gamma": gamma, "reduction": 'mean'} # specific to adapted adapted Kornia implementation 1 (1b in 3/19 notebook)
        # kwargs = {"alpha": weights, "gamma": gamma, "reduction": 'mean'} # testing for/specific to AdeelH implementation
        print(f'focal loss params: \n{kwargs}')
        loss = FocalLoss(**kwargs)
    elif loss_id == 'foc_qwk':
        kwargs = {"n_classes": n_class, "alpha": 1.0 if weights==None else weights, "gamma": gamma, "foc_adjustment": foc_kappa_adjustment}
        print(f'foc_qwk loss params: \n{kwargs}')
        loss = FocalKappaLoss(**kwargs)
    elif loss_id == 'foc_qwk_LC':
        kwargs = {"n_classes": n_class, "alpha": 1.0 if weights==None else weights, "gamma": gamma, "reduction": 'mean', 'foc_coeff': foc_coeff, 'kappa_coeff': kappa_coeff, 'kappa_adjustment': foc_kappa_adjustment}
        print(f'foc_qwk_LC loss params: \n{kwargs}')
        loss = FocalKappaLossLC(**kwargs)
    else:
        raise ValueError("Invalid loss function id. Choices: 'ce', 'mse', 'l1', 'bce', 'qwk', 'coral', 'foc', 'foc_qwk', 'foc_qwk_LC'")

    return loss