""" Module for Casual PINN """

import torch
from .pinn import PINN
from copy import deepcopy
from ..utils import check_consistency
from torch.optim.lr_scheduler import ConstantLR

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732


class CasualPINN(PINN):
    """
    Casual PINN solver class. This class implements Physics Informed Neural
    Network solvers, using a user specified ``model`` to solve a specific
    ``problem``. It can be used for solving both forward and inverse problems.
    Casualiti is enforced during the training step.

    .. seealso:: TODO change

        **Original reference**: Karniadakis, G. E., Kevrekidis, I. G., Lu, L.,
        Perdikaris, P., Wang, S., & Yang, L. (2021).
        Physics-informed machine learning. Nature Reviews Physics, 3(6), 422-440.
        <https://doi.org/10.1038/s42254-021-00314-5>`_.
    """

    def __init__(
        self,
        problem,
        model,
        extra_features=None,
        loss=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam,
        optimizer_kwargs={'lr': 0.001},
        scheduler=ConstantLR,
        scheduler_kwargs={
            "factor": 1,
            "total_iters": 0
        },
        eps=100
    ):
        '''
        :param AbstractProblem problem: The formulation of the problem.
        :param torch.nn.Module model: The neural network model to use.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.nn.Module extra_features: The additional input
            features to use as augmented input.
        :param torch.optim.Optimizer optimizer: The neural network optimizer to
            use; default is :class:`torch.optim.Adam`.
        :param dict optimizer_kwargs: Optimizer constructor keyword args.
        :param torch.optim.LRScheduler scheduler: Learning
            rate scheduler.
        :param dict scheduler_kwargs: LR scheduler constructor keyword args.
        :param int eps: The exponential decay parameter.
        '''
        super().__init__(problem=problem, model=model, extra_features=extra_features,
                         loss=loss, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs,
                         scheduler=scheduler, scheduler_kwargs=scheduler_kwargs)
        
        # checking consistency
        check_consistency(eps, int)
        self._eps = eps

        # initialize weights
        if not hasattr(self.problem, 'temporal_domain'):
            raise ValueError('Casual PINN works only for problems inheritig from TimeDependentProblem.')

        # create a copy of the loss function without reduction
        # self._not_reduced_loss = deepcopy(self.loss)
        # self._not_reduced_loss.reduction = 'none'


    def _split_tensor_into_chunks(self, tensor):
        # extract time tensor
        time_tensor = tensor.extract(self.problem.temporal_domain.variables)
        # sort the time tensors
        _, idx = torch.sort(time_tensor.tensor.flatten())
        tensor = tensor[idx]
        tensor.labels = ['x', 't']
        # find chunks of same time step variable
        chunks = []
        start_idx = 0
        for i in range(1, len(time_tensor)):
            if not torch.equal(time_tensor[i], time_tensor[start_idx]): # if two time variables are not equal
                chunks.append(tensor[start_idx:i]) # create a new chunk
                start_idx = i
        chunks.append(tensor[start_idx:])
        return chunks # return chunks
    
    def _compute_weights(self, loss):
        # compute comulative loss and multiply by epsilos
        cumulative_loss = self._eps * torch.cumsum(loss, dim=0)
        # return the exponential of the weghited negative cumulative sum
        return torch.exp(-cumulative_loss)
    
    def _loss_phys(self, samples, equation):
        # split sequentially ordered time tensors into chunks
        chunks = self._split_tensor_into_chunks(samples)
        # compute residuals - this correspond to ordered loss functions
        # values for each time step. We apply `flatten` such that after
        # concataning the residuals we obtain a tensor of shape #chunks
        time_loss = []
        for chunk in chunks:
            try:
                residual = equation.residual(chunk, self.forward(chunk))
            except TypeError: # this occurs when the function has three inputs, i.e. inverse problem
                residual = equation.residual(chunk, self.forward(chunk), self._params)
            time_loss.append(self.loss(torch.zeros_like(residual), residual))
        # concatenate residuals
        time_loss = torch.stack(time_loss).tensor
        # compute weights (without the gradient storing)
        with torch.no_grad():
            weights = self._compute_weights(time_loss.clone().detach())
        return self.loss(torch.zeros_like(time_loss), weights * time_loss)