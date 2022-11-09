from abc import ABC, abstractmethod
from typing import Any
import ete3

# NOTE: sphinx is currently unable to present this in condensed form, using a string type hint
# of "array-like" in the docstring args for now, instead of ArrayLike hint in call signature
# from numpy.typing import ArrayLike


def init_numpy(use_jax: bool = False):
    r"""Configures the numpy/scipy backend of this module to use the regular or
    JAX version.

    This function is run on import with the default argument (non-JAX).
    """
    global np
    global expit

    if use_jax:
        import jax.numpy as np
        from jax.scipy.special import expit
    else:
        import numpy as np
        from scipy.special import expit


init_numpy()


class Response(ABC):
    r"""Abstract base class for response function mapping
    :py:class:`TreeNode` objects to ``float`` values given parameters."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    @abstractmethod
    def _param_dict(self):
        """Returns a dictionary containing all parameters of the response
        function."""
        pass

    @_param_dict.setter
    @abstractmethod
    def _param_dict(self, d):
        """Configures the parameter values of the response function using the
        provided dictionary (whose format matches that returned by the
        `Response._param_dict` getter method."""
        pass

    @abstractmethod
    def __call__(self, node: "ete3.TreeNode") -> float:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in self.__dict__.items())})"

    def tree_flatten(self):
        items = sorted(self._param_dict.items(), key=lambda item: item[0])
        keys, values = zip(*items)
        return (values, keys)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls()
        obj._param_dict = dict(zip(aux_data, children))
        return obj


class PhenotypeResponse(Response):
    r"""Abstract base class for response function mapping from a
    :py:class:`TreeNode` object's phenotype attribute :math:`x\in\mathbb{R}` to real values given parameters.

    .. math::
        f: \mathbb{R} \to \mathbb{R}

    """

    def __call__(self, node: "ete3.TreeNode") -> float:
        return self.f(node.x)

    @abstractmethod
    def f(self, x) -> float:
        r"""Compute :math:`f(x)`.

        Args:
            x (array-like): Phenotype value.
        """


class ConstantResponse(PhenotypeResponse):
    r"""Returns attribute :math:`\theta\in\mathbb{R}` when an instance is called
    on any :py:class:`TreeNode`.

    Args:
        value: Constant response value.
    """

    def __init__(self, value: float = 1.0):
        self.value = value

    def f(self, x) -> float:
        return self.value * np.ones_like(x)

    @property
    def _param_dict(self):
        return dict(value=self.value)

    @_param_dict.setter
    def _param_dict(self, d):
        self.value = d["value"]


class ExponentialResponse(PhenotypeResponse):
    r"""Exponential response function on a :py:class:`TreeNode` object's
    phenotype attribute :math:`x`.

    .. math::
        f(x) = \theta_3 e^{\theta_1 (x - \theta_2)} + \theta_4

    Args:
        xscale: :math:`\theta_1`
        xshift: :math:`\theta_2`
        yscale: :math:`\theta_3`
        yshift: :math:`\theta_4`
    """

    def __init__(
        self,
        xscale: float = 1.0,
        xshift: float = 0.0,
        yscale: float = 1.0,
        yshift: float = 0.0,
    ):
        self.xscale = xscale
        self.xshift = xshift
        self.yscale = yscale
        self.yshift = yshift

    def f(self, x) -> float:
        return self.yscale * np.exp(self.xscale * (x - self.xshift)) + self.yshift

    @property
    def _param_dict(self):
        return dict(
            xscale=self.xscale,
            xshift=self.xshift,
            yscale=self.yscale,
            yshift=self.yshift,
        )

    @_param_dict.setter
    def _param_dict(self, d):
        self.xscale = d["xscale"]
        self.xshift = d["xshift"]
        self.yscale = d["yscale"]
        self.yshift = d["yshift"]


class SigmoidResponse(PhenotypeResponse):
    r"""Sigmoid response function on a :py:class:`TreeNode` object's phenotype
    attribute :math:`x`.

    .. math::
        f(x) = \frac{\theta_3}{1 + e^{-\theta_1 (x - \theta_2)}} + \theta_4

    Args:
        xscale: :math:`\theta_1`
        xshift: :math:`\theta_2`
        yscale: :math:`\theta_3`
        yshift: :math:`\theta_4`
    """

    def __init__(
        self,
        xscale: float = 1.0,
        xshift: float = 0.0,
        yscale: float = 2.0,
        yshift: float = 0.0,
    ):
        self.xscale = xscale
        self.xshift = xshift
        self.yscale = yscale
        self.yshift = yshift

    def __call__(self, node: "ete3.TreeNode") -> float:
        return self.f(node.x)

    def f(self, x) -> float:
        return self.yscale * expit(self.xscale * (x - self.xshift)) + self.yshift

    @property
    def _param_dict(self):
        return dict(
            xscale=self.xscale,
            xshift=self.xshift,
            yscale=self.yscale,
            yshift=self.yshift,
        )

    @_param_dict.setter
    def _param_dict(self, d):
        self.xscale = d["xscale"]
        self.xshift = d["xshift"]
        self.yscale = d["yscale"]
        self.yshift = d["yshift"]


class SoftReluResponse(PhenotypeResponse):
    r"""Soft ReLU response function on a :py:class:`TreeNode` object's phenotype
    attribute :math:`x`.

    .. math::
        f(x) = \theta_3\log(1 + e^{\theta_1 (x - \theta_2)}) + \theta_4

    Args:
        xscale: :math:`\theta_1`
        xshift: :math:`\theta_2`
        yscale: :math:`\theta_3`
        yshift: :math:`\theta_4`
    """

    def __init__(
        self,
        xscale: float = 1.0,
        xshift: float = 0.0,
        yscale: float = 1.0,
        yshift: float = 0.0,
    ):
        self.xscale = xscale
        self.xshift = xshift
        self.yscale = yscale
        self.yshift = yshift

    def __call__(self, node: "ete3.TreeNode") -> float:
        return self.f(node.x)

    def f(self, x) -> float:
        return (
            self.yscale * np.logaddexp(0, self.xscale * (x - self.xshift)) + self.yshift
        )

    @property
    def _param_dict(self):
        return dict(
            xscale=self.xscale,
            xshift=self.xshift,
            yscale=self.yscale,
            yshift=self.yshift,
        )

    @_param_dict.setter
    def _param_dict(self, d):
        self.xscale = d["xscale"]
        self.xshift = d["xshift"]
        self.yscale = d["yscale"]
        self.yshift = d["yshift"]
