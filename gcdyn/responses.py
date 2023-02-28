r"""
Event rate response functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Abstract base class for defining generic response functions (e.g. :math:`\lambda(x)`, :math:`\mu(x)`, :math:`\gamma(x)`),
with arbitrary :py:class:`TreeNode` attribute dependence.
Some concrete child classes are included.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar
from collections.abc import Callable
import ete3
from jax.tree_util import register_pytree_node

import pandas as pd
import jax.numpy as jnp
import numpy as onp
import jax.scipy.special as jsp
import scipy.special as osp

# NOTE: sphinx is currently unable to present this in condensed form, using a string type hint
# of "array-like" in the docstring args for now, instead of ArrayLike hint in call signature
# from numpy.typing import ArrayLike

from jax.config import config

config.update("jax_enable_x64", True)


class Response(ABC):
    r"""Abstract base class for response function mapping
    :py:class:`TreeNode` objects to ``float`` values given parameters.

    Args:
        grad: Enables JAX compilation and gradient for optimizing.
    """

    @abstractmethod
    def __init__(self, grad: bool = False, *args: Any, **kwargs: Any) -> None:
        self.grad = grad
        self._np = jnp if grad else onp
        self._sp = jsp if grad else osp

    @property
    @abstractmethod
    def _param_dict(self):
        """Returns a dictionary containing all parameters of the response
        function."""

    @_param_dict.setter
    @abstractmethod
    def _param_dict(self, d):
        """Configures the parameter values of the response function using the
        provided dictionary (whose format matches that returned by the
        `Response._param_dict` getter method."""

    @abstractmethod
    def __call__(self, node: ete3.TreeNode) -> float:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in vars(self).items() if not key.startswith('_'))})"


ResponseType = TypeVar("ResponseType", bound=Response)


def _register_with_pytree(response_type: ResponseType) -> None:
    """Registers the `Response` subclass `response_type` as a node in JAX
    pytree, if it is not already.

    This allows parameterized `Response` objects of subclass
    `response_type` to be optimized with JAX.
    """

    def flatten(v):
        # When recreating this object, we need to be able to assign
        # correct values to each parameter, and also be able to set
        # `grad` to its original boolean value

        items = sorted(v._param_dict.items(), key=lambda item: item[0])
        names, values = zip(*items)
        return (values, (names, v.grad))

    def unflatten(aux_data, children):
        new_obj = response_type(grad=aux_data[1])
        new_obj._param_dict = dict(zip(aux_data[0], children))
        return new_obj

    try:
        register_pytree_node(response_type, flatten, unflatten)
    except ValueError:
        # Already registered this type
        pass


class PhenotypeResponse(Response):
    r"""Abstract base class for response function mapping from a
    :py:class:`TreeNode` object's phenotype attribute :math:`x\in\mathbb{R}` to real values given parameters.

    .. math::
        f: \mathbb{R} \to \mathbb{R}

    """

    def __call__(self, node: ete3.TreeNode) -> float:
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
        grad: See :py:class:`Response` docstring.
    """

    def __init__(self, value: float = 1.0, grad: bool = False):
        super().__init__(grad)
        self.value = value

    def f(self, x) -> float:
        return self.value * self._np.ones_like(x)

    @property
    def _param_dict(self) -> dict:
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
        grad: See :py:class:`Response` docstring.
    """

    def __init__(
        self,
        xscale: float = 1.0,
        xshift: float = 0.0,
        yscale: float = 1.0,
        yshift: float = 0.0,
        grad: bool = False,
    ):
        super().__init__(grad)
        self.xscale = xscale
        self.xshift = xshift
        self.yscale = yscale
        self.yshift = yshift

    def f(self, x) -> float:
        return self.yscale * self._np.exp(self.xscale * (x - self.xshift)) + self.yshift

    @property
    def _param_dict(self) -> dict:
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
        grad: See :py:class:`Response` docstring.
    """

    def __init__(
        self,
        xscale: float = 1.0,
        xshift: float = 0.0,
        yscale: float = 2.0,
        yshift: float = 0.0,
        grad: bool = False,
    ):
        super().__init__(grad)
        self.xscale = xscale
        self.xshift = xshift
        self.yscale = yscale
        self.yshift = yshift

    def __call__(self, node: ete3.TreeNode) -> float:
        return self.f(node.x)

    def f(self, x) -> float:
        return (
            self.yscale * self._sp.expit(self.xscale * (x - self.xshift)) + self.yshift
        )

    @property
    def _param_dict(self) -> dict:
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
        grad: See :py:class:`Response` docstring.
    """

    def __init__(
        self,
        xscale: float = 1.0,
        xshift: float = 0.0,
        yscale: float = 1.0,
        yshift: float = 0.0,
        grad: bool = False,
    ):
        super().__init__(grad)
        self.xscale = xscale
        self.xshift = xshift
        self.yscale = yscale
        self.yshift = yshift

    def __call__(self, node: ete3.TreeNode) -> float:
        return self.f(node.x)

    def f(self, x) -> float:
        return (
            self.yscale * self._np.logaddexp(0, self.xscale * (x - self.xshift))
            + self.yshift
        )

    @property
    def _param_dict(self) -> dict:
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


class SequenceResponse(Response):
    r"""Abstract base class for response function mapping from a
    :py:class:`TreeNode` object's sequence attribute to real values given parameters.

    .. math::
        f: \{A,C,G,T\}^n \to \mathbb{R}

    """

    def __call__(self, node: "ete3.TreeNode") -> float:
        return self.f(node.sequence)

    @abstractmethod
    def f(self, sequence) -> float:
        r"""Compute a sequence response.`.

        Args:
            sequence: DNA sequence.
        """


class SequenceContextMutationResponse(SequenceResponse):
    """A Response that accepts a sequence and outputs an aggregate mutation
    rate.

    Importantly, the mutability needs to be in units of mutations per unit time.

    Args:
        mutability: a mapping from local context to mutation rate (mutations per site per unit time)
        seq_to_contexts: a function that accepts a sequence and splits it into local contexts
    """

    def __init__(
        self,
        mutability: pd.Series,
        seq_to_contexts: Callable[str, list[str]],
    ):
        self.mutability = mutability
        self.seq_to_contexts = seq_to_contexts

    @property
    def _param_dict(self) -> dict:
        return {}

    @_param_dict.setter
    def _param_dict(self, d):
        pass

    def f(self, sequence) -> float:
        """The total mutability of a given sequence."""

        contexts = self.seq_to_contexts(sequence)
        return sum(self.mutability[context] for context in contexts)
