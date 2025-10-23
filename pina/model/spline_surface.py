"""Module for the bivariate B-Spline surface model class."""

import torch
from .spline import Spline
from ..utils import check_consistency


class SplineSurface(torch.nn.Module):
    r"""
    The bivariate B-Spline surface model class.

    A bivariate B-spline surface is a parametric surface defined as the tensor
    product of two univariate B-spline curves:

    .. math::

        S(x, y) = \sum_{i,j=1}^{n_x, n_y} B_{i,k}(x) B_{j,s}(y) C_{i,j},
        \quad x \in [x_1, x_m], y \in [y_1, y_l]

    where:

    - :math:`C_{i,j} \in \mathbb{R}^2` are the control points. These fixed
      points influence the shape of the surface but are not generally
      interpolated, except at the boundaries under certain knot multiplicities.
    - :math:`B_{i,k}(x)` and :math:`B_{j,s}(y)` are the B-spline basis functions
      defined over two orthogonal directions, with orders :math:`k` and
      :math:`s`, respectively.
    - :math:`X = \{ x_1, x_2, \dots, x_m \}` and
      :math:`Y = \{ y_1, y_2, \dots, y_l \}` are the non-decreasing knot
      vectors along the two directions.
    """

    def __init__(self, orders, knots_u=None, knots_v=None, control_points=None):
        """
        Initialization of the :class:`SplineSurface` class.

        :param list[int] orders: The orders of the spline along each parametric
            direction. Each order defines the degree of the corresponding basis
            as ``degree = order - 1``.
        :param knots_u: The knots of the spline along the first direction.
            For details on valid formats and initialization modes, see the
            :class:`Spline` class. Default is None.
        :type knots_u: torch.Tensor | dict
        :param knots_v: The knots of the spline along the second direction.
            For details on valid formats and initialization modes, see the
            :class:`Spline` class. Default is None.
        :type knots_v: torch.Tensor | dict
        :param torch.Tensor control_points: The control points defining the
            surface geometry. It must be a two-dimensional tensor of shape
            ``[len(knots_u) - orders[0], len(knots_v) - orders[1]]``.
            If None, they are initialized as learnable parameters with zero
            values. Default is None.
        :raises ValueError: If ``orders`` is not a list of integers.
        :raises ValueError: If ``knots_u`` is neither a torch.Tensor nor a
            dictionary, when provided.
        :raises ValueError: If ``knots_v`` is neither a torch.Tensor nor a
            dictionary, when provided.
        :raises ValueError: If ``control_points`` is not a torch.Tensor,
            when provided.
        :raises ValueError: If ``orders`` is not a list of two elements.
        :raises ValueError: If ``knots_u``, ``knots_v``, and ``control_points``
            are all None.
        """
        super().__init__()

        # Check consistency
        check_consistency(orders, int)
        check_consistency(control_points, (type(None), torch.Tensor))
        check_consistency(knots_u, (type(None), torch.Tensor, dict))
        check_consistency(knots_v, (type(None), torch.Tensor, dict))

        # Check orders is a list of two elements
        if len(orders) != 2:
            raise ValueError("orders must be a list of two elements.")

        # Raise error if neither knots nor control points are provided
        if (knots_u is None or knots_v is None) and control_points is None:
            raise ValueError(
                "control_points cannot be None if knots_u or knots_v is None."
            )

        # Initialize knots_u if not provided
        if knots_u is None and control_points is not None:
            knots_u = {
                "n": control_points.shape[0] + orders[0],
                "min": 0,
                "max": 1,
                "mode": "auto",
            }

        # Initialize knots_v if not provided
        if knots_v is None and control_points is not None:
            knots_v = {
                "n": control_points.shape[1] + orders[1],
                "min": 0,
                "max": 1,
                "mode": "auto",
            }

        # Create two univariate b-splines
        self.spline_u = Spline(order=orders[0], knots=knots_u)
        self.spline_v = Spline(order=orders[1], knots=knots_v)
        self.control_points = control_points

        # Delete unneeded parameters
        delattr(self.spline_u, "_control_points")
        delattr(self.spline_v, "_control_points")

    def forward(self, x):
        """
        Forward pass for the :class:`SplineSurface` model.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        return torch.einsum(
            "...bi, ...bj, ij -> ...b",
            self.spline_u.basis(x.as_subclass(torch.Tensor)[..., 0]),
            self.spline_v.basis(x.as_subclass(torch.Tensor)[..., 1]),
            self.control_points,
        ).unsqueeze(-1)

    @property
    def knots(self):
        """
        The knots of the univariate splines defining the spline surface.

        :return: The knots.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        return self.spline_u.knots, self.spline_v.knots

    @knots.setter
    def knots(self, value):
        """
        Set the knots of the spline surface.

        :param value: A tuple (knots_u, knots_v) containing the knots for both
            parametric directions.
        :type value: tuple(torch.Tensor | dict, torch.Tensor | dict)
        :raises ValueError: If value is not a tuple of two elements.
        """
        # Check value is a tuple of two elements
        if not (isinstance(value, tuple) and len(value) == 2):
            raise ValueError("Knots must be a tuple of two elements.")

        knots_u, knots_v = value
        self.spline_u.knots = knots_u
        self.spline_v.knots = knots_v

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
        Set the control points of the spline surface.

        :param torch.Tensor control_points: The bidimensional control points
            tensor, where each dimension refers to a direction in the parameter
            space. If None, control points are initialized to learnable
            parameters with zero initial value. Default is None.
        :raises ValueError: If in any direction there are not enough knots to
            define the control points, due to the relation:
            #knots = order + #control_points.
        :raises ValueError: If ``control_points`` is not of the correct shape.
        """
        # Save correct shape of control points
        __valid_shape = (
            len(self.spline_u.knots) - self.spline_u.order,
            len(self.spline_v.knots) - self.spline_v.order,
        )

        # If control points are not provided, initialize them
        if control_points is None:

            # Check that there are enough knots to define control points
            if (
                len(self.spline_u.knots) < self.spline_u.order + 1
                or len(self.spline_v.knots) < self.spline_v.order + 1
            ):
                raise ValueError(
                    f"Not enough knots to define control points. Got "
                    f"{len(self.spline_u.knots)} knots along u and "
                    f"{len(self.spline_v.knots)} knots along v, but need at "
                    f"least {self.spline_u.order + 1} and "
                    f"{self.spline_v.order + 1}, respectively."
                )

            # Initialize control points to zero
            control_points = torch.zeros(__valid_shape)

        # Check control points
        if control_points.shape != __valid_shape:
            raise ValueError(
                "control_points must be of the correct shape. ",
                f"Expected {__valid_shape}, got {control_points.shape}.",
            )

        # Register control points as a learnable parameter
        self._control_points = torch.nn.Parameter(
            control_points, requires_grad=True
        )
