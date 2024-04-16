# import torch
# import pytest

# from pina import LabelTensor
# from pina.operators import grad, div, laplacian


# def func_vec(x):
#     return x**2


# def func_scalar(x):
#     x_ = x.extract(['x'])
#     y_ = x.extract(['y'])
#     mu_ = x.extract(['mu'])
#     return x_**2 + y_**2 + mu_**3


# data = torch.rand((5000, 3), requires_grad=True)
# inp = LabelTensor(data, ['x', 'y', 'mu'])
# labels = ['a', 'b', 'c']
# tensor_v = LabelTensor(func_vec(inp), labels)
# tensor_s = LabelTensor(func_scalar(inp).reshape(-1, 1), labels[0])

# # @profile
# def compute_grad_s():
#     # grad scalar
#     grad(tensor_s, inp)

# # @profile
# def compute_grad_v():
#     # grad vector
#     grad(tensor_v, inp)

# # @profile
# def compute_div():
#     # div
#     div(tensor_v, inp)

# # @profile
# def compute_laplacian_s():
#     # laplacian
#     laplacian(tensor_s, inp, components=['a'], d=['x', 'y'])

# # @profile
# def compute_laplacian_v():
#     # laplacian    
#     laplacian(tensor_v, inp)


# compute_grad_s()
# # compute_grad_v()
# # compute_div()
# # compute_laplacian_s()
# # compute_laplacian_v()





""" Simple ODE problem. """


# ===================================================== #
#                                                       #
#  This script implements a simple first order ode.     #
#  The FirstOrderODE class is defined inheriting from   #
#  SpatialProblem. We  denote:                          #
#           y --> field variable                        #
#           x --> spatial variable                      #
#                                                       #
#  The equation is:                                     #
#           dy(x)/dx + y(x) = x                         #
#                                                       #
# ===================================================== #


from pina.problem import SpatialProblem
from pina import Condition
from pina.geometry import CartesianDomain
from pina.operators import grad, fast_grad
from pina.equation import Equation, FixedValue
import torch
from torch.nn import Softplus

from pina.model import FeedForward
from pina.solvers import PINN
from pina.plotter import Plotter
from pina.trainer import Trainer
from torch.func import grad as gradtorch


class FirstOrderODE(SpatialProblem):

    # variable domain range
    x_rng = [0., 5.]
    # field variable
    output_variables = ['y']
    # create domain
    spatial_domain = CartesianDomain({'x': x_rng})

    # define the ode
    def ode(input_, output_):
        y = output_
        x = input_
        return grad(y, x) + y - x

    # define real solution
    def solution(self, input_):
        x = input_
        return x - 1.0 + 2*torch.exp(-x)

    # define problem conditions
    conditions = {
        'BC': Condition(location=CartesianDomain({'x': x_rng[0]}), equation=FixedValue(1.)),
        'D': Condition(location=CartesianDomain({'x': x_rng}), equation=Equation(ode)),
    }

    truth_solution = solution



if __name__ == "__main__":



    # create problem and discretise domain
    problem = FirstOrderODE()
    problem.discretise_domain(n=5000, mode='grid', variables = 'x', locations=['D'])
    problem.discretise_domain(n=1, mode='grid', variables = 'x', locations=['BC'])

    # create model
    model = FeedForward(
        layers=[40, 40],
        output_dimensions=len(problem.output_variables),
        input_dimensions=len(problem.input_variables),
        func=torch.nn.Tanh
    )

    # create solver
    pinn = PINN(
        problem=problem,
        model=model,
        extra_features=None,
        optimizer_kwargs={'lr' : 0.001}
    )

    # create trainer
    trainer = Trainer(solver=pinn, accelerator='cpu', max_epochs=1000)
    print(fast_grad.code)
    trainer.train()