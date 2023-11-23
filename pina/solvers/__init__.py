__all__ = [
    'PINN',
    'CasualPINN',
    'GAROM',
    'SupervisedSolver',
    'SolverInterface'

]

from .garom import GAROM
from .pinn import PINN
from .casual_pinn import CasualPINN
from .supervised import SupervisedSolver
from .solver import SolverInterface
