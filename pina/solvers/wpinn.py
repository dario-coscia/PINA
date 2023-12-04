""" Module for WPINN """

import torch
from torch.optim.lr_scheduler import ConstantLR
from .pinn import PINN
from ..utils import check_consistency


class WeakPINN(PINN):
    """
    WeakPINN solver class. This class implements Weak Physics Informed Neural
    Network solvers, using a user specified ``model`` to solve a specific
    ``problem``. It can be used for solving both forward and inverse problems.

    .. seealso::

        **Original reference**: De Ryck, Tim, Siddhartha Mishra, and Roberto Molinaro.
        "wPINNs: Weak physics informed neural networks for approximating entropy
        solutions of hyperbolic conservation laws."<https://arxiv.org/abs/2207.08483>`_.
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
        '''
        super().__init__(problem=problem, model=model,
                         extra_features=extra_features, loss=loss,
                         optimizer=optimizer, optimizer_kwargs=optimizer_kwargs,
                         scheduler=scheduler, scheduler_kwargs=scheduler_kwargs)
    

