r"""Poisson process responses ^^^^^^^^^^^^^^^^^^^^^^^^^

Abstract base classes for defining generic Poisson processes (e.g.
:math:`\lambda(x, t)`, :math:`\mu(x, t)`, :math:`\gamma(x, t)`), with
arbitrary :py:class:`bdms.TreeNode` attribute dependence. Several
concrete child classes are included.
"""

from __future__ import annotations
from typing import Any, TypeVar, Optional, Union
from typing import TYPE_CHECKING
from collections.abc import Callable
from abc import ABC, abstractmethod
from jax.tree_util import register_pytree_node
from scipy.integrate import quad

import numpy as np
from numpy import random
import scipy as sp


# imports that are only used for type hints
if TYPE_CHECKING:
    from gcdyn import bdms
    import pandas as pd

# NOTE: sphinx is currently unable to present this in condensed form when the sphinx_autodoc_typehints extension is enabled
# TODO: use ArrayLike in various phenotype/time response methods (current float types) once it is available in a stable release
# from numpy.typing import ArrayLike

from jax.config import config

config.update("jax_enable_x64", True)


def set_backend(use_jax=True):
    import numpy
    import scipy
    import jax.numpy
    import jax.scipy.special

    global np, sp
    np = jax.numpy if use_jax else numpy
    sp = jax.scipy.special if use_jax else scipy


class Response(ABC):
    r"""Abstract base class for mapping :py:class:`bdms.TreeNode` objects to a
    Poisson process.
    """

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _register_with_pytree(type(self))

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

    def __call__(self, node: bdms.TreeNode) -> float:
        r"""Call ``self`` to evaluate the Poisson intensity at a tree node.

        Args:
            node: The node whose state is accessed to evaluate the response function.
        """
        return self.λ(node, 0.0)

    @abstractmethod
    def λ(self, node: bdms.TreeNode, Δt: float) -> float:
        r"""Evaluate the Poisson intensity :math:`\lambda(t+\Delta t)` for a
        tree node at time :math:`t`.

        Args:
            node: The node whose state is accessed to evaluate the response function.
            Δt: Time shift from ``node`` at which to evaluate Poisson intensity
                (``0.0`` corresponds to the node's time). This only has an effect if the
                response function is time-inhomogeneous.
        """

    @abstractmethod
    def Λ(self, node: bdms.TreeNode, Δt: float) -> float:
        r"""Evaluate the Poisson intensity measure of the time interval
        :math:`[t, t+\Delta t)`, defined as.

        .. math::
            \Lambda(t, t+Δt) = \int_{t}^{t+\Delta t} \lambda(t')dt',

        for a tree node at time :math:`t`. This is needed for sampling waiting times and evaluating the
        log probability density function of waiting times.

        Args:
            node: The node whose state is accessed to evaluate the response function.
            Δt: Time interval duration (Lebesgue measure).
        """

    @abstractmethod
    def Λ_inv(self, node: bdms.TreeNode, τ: float) -> float:
        r"""Evaluate the inverse function wrt :math:`\Delta t` of :py:meth:`Response.Λ`,
        :math:`\Lambda_t^{-1}(\tau)`, such that :math:`\Lambda_t^{-1}(\Lambda(t, t+\Delta t)) = \Delta t`.
        This is needed for sampling waiting times.
        Note that :math:`\Lambda_t^{-1}` is well-defined iff :math:`\lambda(t) > 0 \forall t`.

        Args:
            node: The node whose state is accessed to evaluate the response function.
            τ: Poisson intensity measure of a time interval.
        """

    def waiting_time_rv(
        self,
        node: bdms.TreeNode,
        rate_multiplier: float = 1.0,
        seed: Optional[Union[int, random.Generator]] = None,
    ) -> float:
        r"""Sample the waiting time :math:`\Delta t` until the first event,
        given the rate process starting at the provided node.

        Args:
            node: The node at which the rate process starts.
            rate_multiplier: A multiplicative factor to apply to the rate process when
                             sampling the waiting time. This can be used to impose a
                             population-size constraint in simulations.
            seed: A seed to initialize the random number generation.
                  If ``None``, then fresh, unpredictable entropy will be pulled from the OS.
                  If an ``int``, then it will be used to derive the initial state.
                  If a :py:class:`numpy.random.Generator`, then it will be used directly.
        """
        if rate_multiplier == 0.0:
            return float("inf")
        rng = random.default_rng(seed)
        return self.Λ_inv(node, rng.exponential(scale=1 / rate_multiplier))

    def waiting_time_logsf(self, node: bdms.TreeNode, Δt: float) -> float:
        r"""Evaluate the logarithm of the survival function of the waiting time
        :math:`\Delta t` given the rate process starting at the provided node.

        Args:
            node: The node at which the rate process starts.
            Δt: The waiting time.
        """
        return -self.Λ(node, Δt)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in vars(self).items() if not key.startswith('_'))})"

    def _flatten(self):
        """Separates out the components of a Response object so its raw
        parameter values can be used in model fitting."""
        items = sorted(self._param_dict.items(), key=lambda item: item[0])
        names, values = zip(*items)
        return type(self), names, values

    @classmethod
    def _from_flat(cls, names, values):
        """Reverses `_flatten`."""
        new_obj = cls()
        new_obj._param_dict = dict(zip(names, values))
        return new_obj


ResponseType = TypeVar("ResponseType", bound=Response)


def _register_with_pytree(response_type: ResponseType) -> None:
    r"""Registers the `Response` subclass `response_type` as a node in JAX
    pytree, if it is not already.

    This allows parameterized `Response` objects of subclass
    `response_type` to be optimized with JAX.
    """

    def flatten(response):
        _, names, values = response._flatten()
        return values, names

    def unflatten(cls, static_data, dynamic_data):
        return cls._from_flat(static_data, dynamic_data)

    try:
        register_pytree_node(response_type, flatten, response_type._from_flat)
    except ValueError:
        # Already registered this type
        pass


class HomogeneousResponse(Response):
    r"""Abstract base class for response functions mapping
    :py:class:`bdms.TreeNode` objects to a homogenous Poisson process.

    Args:
        grad: Enables JAX compilation and gradient for optimizing.
    """

    @abstractmethod
    def λ_homogeneous(self, node: bdms.TreeNode) -> float:
        r"""Evaluate the homogeneous Poisson intensity :math:`\lambda` for a
        tree node.

        Args:
            node: The node whose state is accessed to evaluate the response function.
        """

    def λ(self, node: bdms.TreeNode, Δt: float) -> float:
        return self.λ_homogeneous(node)

    def Λ(self, node: bdms.TreeNode, Δt: float) -> float:
        return self.λ_homogeneous(node) * Δt

    def Λ_inv(self, node: bdms.TreeNode, τ: float) -> float:
        return τ / self.λ_homogeneous(node)


class PhenotypeResponse(HomogeneousResponse):
    r"""Abstract base class for response function mapping from a
    :py:class:`bdms.TreeNode` object's phenotype attribute
    :math:`x\in\mathbb{R}` to a homogeneous Poisson process.
    """

    def λ_homogeneous(self, node: bdms.TreeNode) -> float:
        return self.λ_phenotype(node.x)

    @abstractmethod
    def λ_phenotype(self, x: float) -> float:
        r"""Evaluate the Poisson intensity :math:`\lambda_x` for phenotype
        :math:`x`.

        Args:
            x: Phenotype value.
        """


class ConstantResponse(PhenotypeResponse):
    r"""Returns attribute :math:`\theta\in\mathbb{R}` when an instance is called
    on any :py:class:`bdms.TreeNode`.

    Args:
        value: Constant response value.
    """

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.value = value

    def λ_phenotype(self, x: float) -> float:
        return self.value * np.ones_like(x)

    @property
    def _param_dict(self) -> dict:
        return dict(value=self.value)

    @_param_dict.setter
    def _param_dict(self, d):
        self.value = d["value"]


class ExponentialResponse(PhenotypeResponse):
    r"""Exponential response function on a :py:class:`bdms.TreeNode` object's
    phenotype attribute :math:`x`.

    .. math::
        \lambda_x = \theta_3 e^{\theta_1 (x - \theta_2)} + \theta_4

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
        super().__init__()
        self.xscale = xscale
        self.xshift = xshift
        self.yscale = yscale
        self.yshift = yshift

    def λ_phenotype(self, x: float) -> float:
        x = np.asarray(x)
        return self.yscale * np.exp(self.xscale * (x - self.xshift)) + self.yshift

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
    r"""Sigmoid response function on a :py:class:`bdms.TreeNode` object's
    phenotype attribute :math:`x`.

    .. math::
        \lambda_x = \frac{\theta_3}{1 + e^{-\theta_1 (x - \theta_2)}} + \theta_4

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
        super().__init__()
        self.xscale = xscale
        self.xshift = xshift
        self.yscale = yscale
        self.yshift = yshift

    def λ_phenotype(self, x: float) -> float:
        x = np.asarray(x)
        return self.yscale * sp.expit(self.xscale * (x - self.xshift)) + self.yshift

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
    r"""Soft ReLU response function on a :py:class:`bdms.TreeNode` object's
    phenotype attribute :math:`x`.

    .. math::
        \lambda_x = \theta_3\log(1 + e^{\theta_1 (x - \theta_2)}) + \theta_4

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
        super().__init__()
        self.xscale = xscale
        self.xshift = xshift
        self.yscale = yscale
        self.yshift = yshift

    def λ_phenotype(self, x: float) -> float:
        x = np.asarray(x)
        return (
            self.yscale * np.logaddexp(0, self.xscale * (x - self.xshift)) + self.yshift
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


class SequenceContextMutationResponse(HomogeneousResponse):
    r"""A Response that accepts a sequence and outputs an aggregate mutation
    rate.

    The mutability needs to be in units of mutations per unit time.

    A ``sequence_context`` node feature is created when nodes are first operated on.

    Args:
        mutability: a mapping from local context to mutation rate (mutations per site per unit time)
        mutation_intensity: a scaling factor for the mutability
    """

    def __init__(
        self,
        mutability: pd.Series,
        mutation_intensity: float = 1.0,
    ):
        self.mutability = (mutation_intensity * mutability).to_dict()

    @property
    def _param_dict(self) -> dict:
        return dict(mutation_intensity=self.mutation_intensity)

    @_param_dict.setter
    def _param_dict(self, d):
        self.mutation_intensity = d["mutation_intensity"]

    def λ_homogeneous(self, node: bdms.TreeNode) -> float:
        return sum(self.mutability[context] for context in node.sequence_context)


class PhenotypeTimeResponse(Response):
    r"""Abstract base class for response function mapping from a
    :py:class:`bdms.TreeNode` object's phenotype attribute
    :math:`x\in\mathbb{R}` and time :math:`t\in\mathbb{R}_{\ge 0}` to a Poisson
    process. Explicit phenotype and time dependence must be specified by
    concrete subclasses.

    This abstract base class provides generic default implementations of
    :py:meth:`Λ` and :py:meth:`Λ_inv` via quadrature and root-finding,
    respectively.

    Args:
        tol: Tolerance for root-finding.
        maxiter: Maximum number of iterations for root-finding.
    """

    def __init__(self, tol: float = 1e-6, maxiter: int = 100, *args, **kwargs):
        # if grad:
        #     raise NotImplementedError(f"gradients are not implemented for {type(self)}")
        self.tol = tol
        self.maxiter = maxiter

    @abstractmethod
    def λ_phenotype_time(self, x: float, t: float) -> float:
        r"""Evaluate the Poisson intensity :math:`\lambda_x(t)` for phenotype
        :math:`x` at time :math:`t`.

        Args:
            x: Phenotype.
            t: Time.
        """

    def λ(self, node: bdms.TreeNode, Δt: float) -> float:
        return self.λ_phenotype_time(node.x, node.t + Δt)

    def Λ(self, node: bdms.TreeNode, Δt: float) -> float:
        return quad(lambda Δt: self.λ(node, Δt), 0, Δt, limit=1000)[0]

    def Λ_inv(self, node: bdms.TreeNode, τ: float) -> float:
        # initial guess via rate at node
        Δt = τ / self(node)
        # non-negative Newton-Raphson root-finding
        converged = False
        for iter in range(self.maxiter):
            if self.λ(node, Δt) == 0:
                return np.inf
            with np.errstate(over="ignore"):
                Δt = max(Δt - (self.Λ(node, Δt) - τ) / self.λ(node, Δt), 0.0)
            if abs(self.Λ(node, Δt) - τ) < self.tol:
                converged = True
                break
        if not converged:
            print(f"Δt={Δt}, Λ={self.Λ(node, Δt)}, τ={τ}, λ={self.λ(node, Δt)}")
            raise RuntimeError(
                f"Newton-Raphson failed to converge after {self.maxiter} iterations with Δt={Δt} and error={abs(self.Λ(node, Δt) - τ):.3f}"
            )
        return Δt


class ModulatedPhenotypeResponse(PhenotypeTimeResponse):
    r"""An inhomogeneous phenotype response that combines a homogeneous
    phenotype response with a time-dependent external field :math:`f(t)` that
    modulates the effective phenotype via an interaction :math:`\tilde x = \phi(x, f(t))`
    that maps the phenotype and external field to the effective phenotype.
    For example, if :math:`\phi(x, f(t)) = x - f(t)`, then the external field
    represents an additive phenotype shift.
    The homogeneous phenotype response is evaluated at the effective phenotype.

    Args:
        phenotype_response: a homogeneous phenotype response for the effective phenotype :math:`x - f(t)`.
        external_field: external field :math:`f(t)`, a function that maps time to the external field.
        interaction: a function :math:`\phi(x, f(t))` that maps the phenotype and external field to the effective phenotype.
        tol: See :py:class:`PhenotypeTimeResponse`.
        maxiter: See :py:class:`PhenotypeTimeResponse`.
    """

    def __init__(
        self,
        phenotype_response: PhenotypeResponse,
        external_field: Callable[[float], float],
        interaction: Callable[[float, float], float] = lambda x, f: x - f,
        tol: float = 1e-6,
        maxiter: int = 100,
    ):
        self.phenotype_response = phenotype_response
        self.external_field = external_field
        self.interaction = interaction
        self.tol = tol
        self.maxiter = maxiter

    def λ_phenotype_time(self, x: float, t: float) -> float:
        effective_phenotype = self.interaction(x, self.external_field(t))
        return self.phenotype_response.λ_phenotype(effective_phenotype)

    @property
    def _param_dict(self) -> dict:
        return self.phenotype_response._param_dict

    @_param_dict.setter
    def _param_dict(self, d):
        self.phenotype_response._param_dict = d


class ModulatedRateResponse(PhenotypeTimeResponse):
    r"""An inhomogeneous phenotype response that modulates a homogeneous
    phenotype response rate :math:`\lambda_x` via a time-dependent function
    :math:`\tilde\lambda(\lambda_x, x, t)` to yield a time-dependent modulated rate.

    Args:
        phenotype_response: a homogeneous phenotype response.
        modulation: a function :math:`\tilde\lambda(\lambda_x, x, t)` that maps the original rate :math:`\lambda_x`, phenotype :math:`x`, and time :math:`t` to the modulated rate.
        tol: See :py:class:`PhenotypeTimeResponse`.
        maxiter: See :py:class:`PhenotypeTimeResponse`.
    """

    def __init__(
        self,
        phenotype_response: PhenotypeResponse,
        modulation: Callable[[float, float, float], float],
        tol: float = 1e-6,
        maxiter: int = 100,
    ):
        self.phenotype_response = phenotype_response
        self.modulation = modulation
        self.tol = tol
        self.maxiter = maxiter

    def λ_phenotype_time(self, x: float, t: float) -> float:
        return self.modulation(self.phenotype_response.λ_phenotype(x), x, t)

    @property
    def _param_dict(self) -> dict:
        return self.phenotype_response._param_dict

    @_param_dict.setter
    def _param_dict(self, d):
        self.phenotype_response._param_dict = d
