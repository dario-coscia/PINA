__all__ = [
    'PINN',
    'WeakPINN',
    'GAROM',
    'SupervisedSolver',
    'SolverInterface'

]

from .garom import GAROM
from .pinn import PINN
from .wpinn import WeakPINN
from .supervised import SupervisedSolver
from .solver import SolverInterface
