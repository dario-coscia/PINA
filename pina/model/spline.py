"""Module for the B-Spline model class."""

import warnings
import torch
from ..utils import check_positive_integer, check_consistency


class Spline(torch.nn.Module):
    r"""
    The univariate B-Spline curve model class.

    A univariate B-spline curve of order :math:`k` is a parametric curve defined
    as a linear combination of B-spline basis functions and control points:

    .. math::

        S(x) = \sum_{i=1}^{n} B_{i,k}(x) C_i, \quad x \in [x_1, x_m]

    where:

    - :math:`C_i \in \mathbb{R}` are the control points. These fixed points
      influence the shape of the curve but are not generally interpolated,
      except at the boundaries under certain knot multiplicities.
    - :math:`B_{i,k}(x)` are the B-spline basis functions of order :math:`k`,
      i.e., piecewise polynomials of degree :math:`k-1` with support on the
      interval :math:`[x_i, x_{i+k}]`.
    - :math:`X = \{ x_1, x_2, \dots, x_m \}` is the non-decreasing knot vector.

    If the first and last knots are repeated :math:`k` times, then the curve
    interpolates the first and last control points.


    .. note::

        The curve is forced to be zero outside the interval defined by the
        first and last knots.


    :Example:

    >>> from pina.model import Spline
    >>> import torch

    >>> knots1 = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    >>> spline1 = Spline(order=3, knots=knots1, control_points=None)

    >>> knots2 = {"n": 7, "min": 0.0, "max": 2.0, "mode": "auto"}
    >>> spline2 = Spline(order=3, knots=knots2, control_points=None)

    >>> knots3 = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    >>> control_points3 = torch.tensor([0.0, 1.0, 3.0, 2.0])
    >>> spline3 = Spline(order=3, knots=knots3, control_points=control_points3)
    """

    def __init__(self, order=4, knots=None, control_points=None):
        """
        Initialization of the :class:`Spline` class.

        :param int order: The order of the spline. The corresponding basis
            functions are polynomials of degree ``order - 1``. Default is 4.
        :param knots: The knots of the spline. If a tensor is provided, knots
            are set directly from the tensor. If a dictionary is provided, it
            must contain the keys ``"n"``, ``"min"``, ``"max"``, and ``"mode"``.
            Here, ``"n"`` specifies the number of knots, ``"min"`` and ``"max"``
            define the interval, and ``"mode"`` selects the sampling strategy.
            The supported modes are ``"uniform"``, where the knots are evenly
            spaced over :math:`[min, max]`, and ``"auto"``, where knots are
            constructed to ensure that the spline interpolates the first and
            last control points. In this case, the number of knots is adjusted
            if :math:`n < 2 * order`. If None is given, knots are initialized
            automatically over :math:`[0, 1]` ensuring interpolation of the
            first and last control points. Default is None.
        :type knots: torch.Tensor | dict
        :param torch.Tensor control_points: The control points of the spline.
            If None, they are initialized as learnable parameters with an
            initial value of zero. Default is None.
        :raises AssertionError: If ``order`` is not a positive integer.
        :raises ValueError: If ``knots`` is neither a torch.Tensor nor a
            dictionary, when provided.
        :raises ValueError: If ``control_points`` is not a torch.Tensor,
            when provided.
        :raises ValueError: If both ``knots`` and ``control_points`` are None.
        :raises ValueError: If ``knots`` is not one-dimensional.
        :raises ValueError: If ``control_points`` is not one-dimensional.
        :raises ValueError: If the number of ``knots`` is not equal to the sum
            of ``order`` and the number of ``control_points.``
        :raises UserWarning: If the number of control points is lower than the
            order, resulting in a degenerate spline.
        """
        super().__init__()

        # Check consistency
        check_positive_integer(value=order, strict=True)
        check_consistency(knots, (type(None), torch.Tensor, dict))
        check_consistency(control_points, (type(None), torch.Tensor))

        # Raise error if neither knots nor control points are provided
        if knots is None and control_points is None:
            raise ValueError("knots and control_points cannot both be None.")

        # Initialize knots if not provided
        if knots is None and control_points is not None:
            knots = {
                "n": len(control_points) + order,
                "min": 0,
                "max": 1,
                "mode": "auto",
            }

        # Initialization - knots and control points managed by their setters
        self.order = order
        self.knots = knots
        self.control_points = control_points

        # Check dimensionality of knots
        if self.knots.ndim > 1:
            raise ValueError("knots must be one-dimensional.")

        # Check dimensionality of control points
        if self.control_points.ndim > 1:
            raise ValueError("control_points must be one-dimensional.")

        # Raise error if #knots != order + #control_points
        if len(self.knots) != self.order + len(self.control_points):
            raise ValueError(
                f" The number of knots must be equal to order + number of"
                f" control points. Got {len(self.knots)} knots, {self.order}"
                f" order and {len(self.control_points)} control points."
            )

        # Raise warning if spline is degenerate
        if len(self.control_points) < self.order:
            warnings.warn(
                "The number of control points is smaller than the spline order."
                " This creates a degenerate spline with limited flexibility.",
                UserWarning,
            )

        # Precompute boundary interval index
        self._boundary_interval_idx = self._compute_boundary_interval()

    def _compute_boundary_interval(self):
        """
        Precompute the index of the rightmost non-degenerate interval to improve
        performance, eliminating the need to perform a search loop in the basis
        function on each call.

        :return: The index of the rightmost non-degenerate interval.
        :rtype: int
        """
        # Return 0 if there is a single interval
        if len(self.knots) < 2:
            return 0

        # Find all indices where knots are strictly increasing
        diffs = self.knots[1:] - self.knots[:-1]
        valid = torch.nonzero(diffs > 0, as_tuple=False)

        # If all knots are equal, return 0 for degenerate spline
        if valid.numel() == 0:
            return 0

        # Otherwise, return the last valid index
        return int(valid[-1])

    def basis(self, x):
        """
        Compute the basis functions for the spline using an iterative approach.
        This is a vectorized implementation based on the Cox-de Boor recursion.

        :param torch.Tensor x: The points to be evaluated.
        :return: The basis functions evaluated at x.
        :rtype: torch.Tensor
        """
        # Add a final dimension to x
        x = x.unsqueeze(-1)

        # Add an initial dimension to knots
        knots = self.knots.unsqueeze(0)

        # Base case of recursion: indicator functions for the intervals
        basis = (x >= knots[..., :-1]) & (x < knots[..., 1:])
        basis = basis.to(x.dtype)

        # One-dimensional knots case: ensure rightmost boundary inclusion
        if self._boundary_interval_idx is not None:

            # Extract left and right knots of the rightmost interval
            knot_left = knots[..., self._boundary_interval_idx]
            knot_right = knots[..., self._boundary_interval_idx + 1]

            # Identify points at the rightmost boundary
            at_rightmost_boundary = (
                x.squeeze(-1) >= knot_left
            ) & torch.isclose(x.squeeze(-1), knot_right, rtol=1e-8, atol=1e-10)

            # Ensure the correct value is set at the rightmost boundary
            if torch.any(at_rightmost_boundary):
                basis[..., self._boundary_interval_idx] = torch.logical_or(
                    basis[..., self._boundary_interval_idx].bool(),
                    at_rightmost_boundary,
                ).to(basis.dtype)

        # Iterative case of recursion
        for i in range(1, self.order):

            # Compute the denominators for both terms
            denom1 = knots[..., i:-1] - knots[..., : -(i + 1)]
            denom2 = knots[..., i + 1 :] - knots[..., 1:-i]

            # Ensure no division by zero
            denom1 = torch.where(
                torch.abs(denom1) < 1e-8, torch.ones_like(denom1), denom1
            )
            denom2 = torch.where(
                torch.abs(denom2) < 1e-8, torch.ones_like(denom2), denom2
            )

            # Compute the two terms of the recursion
            term1 = ((x - knots[..., : -(i + 1)]) / denom1) * basis[..., :-1]
            term2 = ((knots[..., i + 1 :] - x) / denom2) * basis[..., 1:]

            # Combine terms to get the new basis
            basis = term1 + term2

        return basis

    def forward(self, x):
        """
        Forward pass for the :class:`Spline` model.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        return torch.einsum(
            "...bi, i -> ...b",
            self.basis(x.as_subclass(torch.Tensor)).squeeze(-1),
            self.control_points,
        )

    @property
    def control_points(self):
        """
        The control points of the spline.

        :return: The control points.
        :rtype: torch.Tensor
        """
        return self._control_points

    @control_points.setter
    def control_points(self, control_points):
        """
        Set the control points of the spline.

        :param torch.Tensor control_points: The control points tensor. If None,
            control points are initialized to learnable parameters with zero
            initial value. Default is None.
        :raises ValueError: If there are not enough knots to define the control
            points, due to the relation: #knots = order + #control_points.
        """
        # If control points are not provided, initialize them
        if control_points is None:

            # Check that there are enough knots to define control points
            if len(self.knots) < self.order + 1:
                raise ValueError(
                    f"Not enough knots to define control points. Got "
                    f"{len(self.knots)} knots, but need at least "
                    f"{self.order + 1}."
                )

            # Initialize control points to zero
            control_points = torch.zeros(len(self.knots) - self.order)

        # Set control points
        self._control_points = torch.nn.Parameter(
            control_points, requires_grad=True
        )

    @property
    def knots(self):
        """
        The knots of the spline.

        :return: The knots.
        :rtype: torch.Tensor
        """
        return self._knots

    @knots.setter
    def knots(self, value):
        """
        Set the knots of the spline.

        :param value: The knots of the spline. If a tensor is provided, knots
            are set directly from the tensor. If a dictionary is provided, it
            must contain the keys ``"n"``, ``"min"``, ``"max"``, and ``"mode"``.
            Here, ``"n"`` specifies the number of knots, ``"min"`` and ``"max"``
            define the interval, and ``"mode"`` selects the sampling strategy.
            The supported modes are ``"uniform"``, where the knots are evenly
            spaced over :math:`[min, max]`, and ``"auto"``, where knots are
            constructed to ensure that the spline interpolates the first and
            last control points. In this case, the number of knots is inferred
            and the ``"n"`` key is ignored.
        :type value: torch.Tensor | dict
        :raises ValueError: If a dictionary is provided but does not contain
            the required keys.
        :raises ValueError: If the mode specified in the dictionary is invalid.
        """
        # If a dictionary is provided, initialize knots accordingly
        if isinstance(value, dict):

            # Check that required keys are present
            required_keys = {"n", "min", "max", "mode"}
            if not required_keys.issubset(value.keys()):
                raise ValueError(
                    f"When providing knots as a dictionary, the following "
                    f"keys must be present: {required_keys}. Got "
                    f"{value.keys()}."
                )

            # Uniform sampling of knots
            if value["mode"] == "uniform":
                value = torch.linspace(value["min"], value["max"], value["n"])

            # Automatic sampling of interpolating knots
            elif value["mode"] == "auto":

                # Repeat the first and last knots 'order' times
                initial_knots = torch.ones(self.order) * value["min"]
                final_knots = torch.ones(self.order) * value["max"]

                # Number of internal knots
                n_internal = value["n"] - 2 * self.order

                # If no internal knots are needed, just concatenate boundaries
                if n_internal <= 0:
                    value = torch.cat((initial_knots, final_knots))

                # Else, sample internal knots uniformly and exclude boundaries
                # Recover the correct number of internal knots when slicing by
                # adding 2 to n_internal
                else:
                    internal_knots = torch.linspace(
                        value["min"], value["max"], n_internal + 2
                    )[1:-1]
                    value = torch.cat(
                        (initial_knots, internal_knots, final_knots)
                    )

            # Raise error if mode is invalid
            else:
                raise ValueError(
                    f"Invalid mode for knots initialization. Got "
                    f"{value['mode']}, but expected 'uniform' or 'auto'."
                )

        # Set knots
        self.register_buffer("_knots", value.sort(dim=0).values)

        # Recompute boundary interval when knots change
        if hasattr(self, "_boundary_interval_idx"):
            self._boundary_interval_idx = self._compute_boundary_interval()
