"""
Module for operators vectorize implementation. Differential operators are used to write any differential problem.
These operators are implemented to work on different accellerators: CPU, GPU, TPU or MPS.
All operators take as input a tensor onto which computing the operator, a tensor with respect
to which computing the operator, the name of the output variables to calculate the operator
for (in case of multidimensional functions), and the variables name on which the operator is calculated.
"""

import torch

from pina.label_tensor import LabelTensor
from typing import List, Optional


from torch.jit.annotations import List as TorchList
from torch.jit.annotations import TensorType

@torch.jit.script
def fast_grad(y: torch.Tensor, x: torch.Tensor) -> Optional[torch.Tensor]:
    grad_outputs : List[Optional[torch.Tensor]] = [ torch.ones_like(y) ]
    grad = torch.autograd.grad([y,], [x], grad_outputs=grad_outputs, create_graph=True, allow_unused=True, retain_graph=True)
    
    # optional type refinement using an if statement
    if grad is not None:
        grad = grad[0]
        
    return grad # grad can be None here, so it is Optional[torch.Tensor]


# @profile
def grad(output_, input_, components = None, d = None):
    """
    Perform gradient operation. The operator works for vectorial and scalar
    functions, with multiple input coordinates.

    :param LabelTensor output_: the output tensor onto which computing the
        gradient.
    :param LabelTensor input_: the input tensor with respect to which computing
        the gradient.
    :param list(str) components: the name of the output variables to calculate
        the gradient for. It should be a subset of the output labels. If None,
        all the output variables are considered. Default is None.
    :param list(str) d: the name of the input variables on which the gradient is
        calculated. d should be a subset of the input labels. If None, all the
        input variables are considered. Default is None.

    :return: the gradient tensor.
    :rtype: LabelTensor
    """
    # @profile
    def grad_scalar_output(output_, input_, d):
        """
        Perform gradient operation for a scalar output.

        :param LabelTensor output_: the output tensor onto which computing the
            gradient. It has to be a column tensor.
        :param LabelTensor input_: the input tensor with respect to which
            computing the gradient.
        :param list(str) d: the name of the input variables on which the
            gradient is calculated. d should be a subset of the input labels. If
            None, all the input variables are considered. Default is None.

        :raises RuntimeError: a vectorial function is passed.
        :raises RuntimeError: missing derivative labels.
        :return: the gradient tensor.
        :rtype: LabelTensor
        """

        if len(output_.labels) != 1:
            raise RuntimeError("only scalar function can be differentiated")
        if not all([di in input_.labels for di in d]):
            raise RuntimeError("derivative labels missing from input tensor")

        output_fieldname = output_.labels[0]
        # gradients = fast_grad(output_, input_)
        # gradients = gradients.as_subclass(LabelTensor)
        gradients = torch.autograd.grad(
            output_,
            input_,
            grad_outputs=torch.ones(
                output_.size(), dtype=output_.dtype, device=output_.device
            ),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        gradients.labels = input_.labels
        gradients = gradients.extract(d)
        gradients.labels = [f"d{output_fieldname}d{i}" for i in d]

        return gradients

    if not isinstance(input_, LabelTensor):
        raise TypeError

    if d is None:
        d = input_.labels

    if components is None:
        components = output_.labels

    if output_.shape[1] == 1:  # scalar output ################################

        if components != output_.labels:
            raise RuntimeError
        gradients = grad_scalar_output(output_, input_, d)

    elif output_.shape[1] >= 2:  # vector output ##############################

        for i, c in enumerate(components):
            c_output = output_.extract([c])
            if i == 0:
                gradients = grad_scalar_output(c_output, input_, d)
            else:
                gradients = gradients.append(
                    grad_scalar_output(c_output, input_, d)
                )
    else:
        raise NotImplementedError

    return gradients

# @profile
def div(output_, input_, components=None, d=None):
    """
    Perform divergence operation. The operator works for vectorial functions,
    with multiple input coordinates.

    :param LabelTensor output_: the output tensor onto which computing the
        divergence.
    :param LabelTensor input_: the input tensor with respect to which computing
        the divergence.
    :param list(str) components: the name of the output variables to calculate
        the divergence for. It should be a subset of the output labels. If None,
        all the output variables are considered. Default is None.
    :param list(str) d: the name of the input variables on which the divergence
        is calculated. d should be a subset of the input labels. If None, all
        the input variables are considered. Default is None.

    :raises TypeError: div operator works only for LabelTensor.
    :raises ValueError: div operator works only for vector fields.
    :raises ValueError: div operator must derive all components with
        respect to all coordinates.
    :return: the divergenge tensor.
    :rtype: LabelTensor
    """
    if not isinstance(input_, LabelTensor):
        raise TypeError

    if d is None:
        d = input_.labels

    if components is None:
        components = output_.labels

    if output_.shape[1] < 2 or len(components) < 2:
        raise ValueError("div supported only for vector fields")

    if len(components) != len(d):
        raise ValueError

    grad_output = grad(output_, input_, components, d)
    div = torch.zeros(input_.shape[0], 1, device=output_.device)
    labels = [None] * len(components)
    for i, (c, d) in enumerate(zip(components, d)):
        c_fields = f"d{c}d{d}"
        div[:, 0] += grad_output.extract(c_fields).sum(axis=1)
        labels[i] = c_fields

    div = div.as_subclass(LabelTensor)
    div.labels = ["+".join(labels)]
    return div

# @promfile
def laplacian(output_, input_, components=None, d=None, method="std"):
    """
    Compute Laplace operator. The operator works for vectorial and
    scalar functions, with multiple input coordinates.

    :param LabelTensor output_: the output tensor onto which computing the
        Laplacian.
    :param LabelTensor input_: the input tensor with respect to which computing
        the Laplacian.
    :param list(str) components: the name of the output variables to calculate
        the Laplacian for. It should be a subset of the output labels. If None,
        all the output variables are considered. Default is None.
    :param list(str) d: the name of the input variables on which the Laplacian
        is calculated. d should be a subset of the input labels. If None, all
        the input variables are considered. Default is None.
    :param str method: used method to calculate Laplacian, defaults to 'std'.
    :raises ValueError: for vectorial field derivative with respect to
        all coordinates must be performed.
    :raises NotImplementedError: 'divgrad' not implemented as method.
    :return: The tensor containing the result of the Laplacian operator.
    :rtype: LabelTensor
    """
    if d is None:
        d = input_.labels

    if components is None:
        components = output_.labels

    if len(components) != len(d) and len(components) != 1:
        raise ValueError

    if method == "divgrad":
        raise NotImplementedError("divgrad not implemented as method")
        # TODO fix
        # grad_output = grad(output_, input_, components, d)
        # result = div(grad_output, input_, d=d)
    elif method == "std":

        if len(components) == 1:
            grad_output = grad(output_, input_, components=components, d=d)
            result = torch.zeros(output_.shape[0], 1, device=output_.device)
            for i, label in enumerate(grad_output.labels):
                gg = grad(grad_output, input_, d=d, components=[label])
                result[:, 0] += super(torch.Tensor, gg.T).__getitem__(
                    i
                )  # TODO improve
            labels = [f"dd{components[0]}"]

        else:
            result = torch.empty(
                input_.shape[0], len(components), device=output_.device
            )
            labels = [None] * len(components)
            for idx, (ci, di) in enumerate(zip(components, d)):

                if not isinstance(ci, list):
                    ci = [ci]
                if not isinstance(di, list):
                    di = [di]

                grad_output = grad(output_, input_, components=ci, d=di)
                result[:, idx] = grad(grad_output, input_, d=di).flatten()
                labels[idx] = f"dd{ci}dd{di}"

    result = result.as_subclass(LabelTensor)
    result.labels = labels
    return result

# @profile
def advection(output_, input_, velocity_field, components=None, d=None):
    """
    Perform advection operation. The operator works for vectorial functions,
    with multiple input coordinates.

    :param LabelTensor output_: the output tensor onto which computing the
        advection.
    :param LabelTensor input_: the input tensor with respect to which computing
        the advection.
    :param str velocity_field: the name of the output variables which is used
        as velocity field. It should be a subset of the output labels.
    :param list(str) components: the name of the output variables to calculate
        the Laplacian for. It should be a subset of the output labels. If None,
        all the output variables are considered. Default is None.
    :param list(str) d: the name of the input variables on which the advection
        is calculated. d should be a subset of the input labels. If None, all
        the input variables are considered. Default is None.
    :return: the tensor containing the result of the advection operator.
    :rtype: LabelTensor
    """
    if d is None:
        d = input_.labels

    if components is None:
        components = output_.labels

    tmp = (
        grad(output_, input_, components, d)
        .reshape(-1, len(components), len(d))
        .transpose(0, 1)
    )

    tmp *= output_.extract(velocity_field)
    return tmp.sum(dim=2).T
