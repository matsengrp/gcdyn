r"""Birth-death-mutation-sampling (BDMS) process.

The BDMS process is defined by the following parameters:

* :math:`\Delta t`: run time of the process
* :math:`\lambda(x)`: birth rate of phenotype :math:`x`
* :math:`\mu(x)`: death rate of phenotype :math:`x`
* :math:`\gamma(x)`: mutation rate of phenotype :math:`x`
* :math:`\mathcal{p}(x\mid x')`: phenotypic mutation transition density conditional on initial phenotype :math:`x'`,
  which we draw from to generate mutation effects when a mutation event occurs
* :math:`\rho`: sampling probability of surviving lineages after :math:`\Delta t`

Primary class :py:class:`TreeNode`
----------------------------------

This module's primary class :py:class:`TreeNode` subclasses ETE's :py:class:`ete3.TreeNode`,
with these notable differences:

* Attribute :py:attr:`t`: the time :math:`t \in \mathbb{R}_{\ge 0}` of the event at the node
* Attribute :py:attr:`x`: the phenotype :math:`x \in \mathbb{R}` of the node
* Attribute :py:attr:`event`: the event that occurred at the node
* Attribute :py:attr:`n_mutations`: the number of mutations that occurred on the branch above the node
* Method :py:meth:`TreeNode.evolve`: evolve the tree, adding nodes according a BDMS process
* Method :py:meth:`TreeNode.sample_survivors`: sample a subset of surviving leaves from the tree
* Method :py:meth:`TreeNode.prune`: prune the tree subtree induced by the sampled leaves
  (overrides ETE's :py:meth:`ete3.TreeNode.prune`)
* Method :py:meth:`TreeNode.render`: visualizes the tree (overrides ETE's :py:meth:`ete3.TreeNode.render`)

Additional classes:
-------------------

Rate response functions :py:class:`Response`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Abstract base class for defining generic response functions (i.e. :math:`\lambda(x)`, :math:`\mu(x)`, :math:`\gamma(x)`),
with arbitrary :py:class:`TreeNode` attribute dependence.
Some concrete child classes are included.

Mutation effects generators :py:class:`Mutator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Abstract base class for defining generic mutation effect generators (i.e. :math:`\mathcal{p}(x\mid x')`),
with arbitrary :py:class:`TreeNode` attribute dependence.
Some concrete child classes are included.
"""
from abc import ABC, abstractmethod
import ete3
from ete3.coretype.tree import TreeError
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Any, Optional, Union

# NOTE: sphinx is currently unable to present this in condensed form, using a string type hint
# of "array-like" in the docstring args for now, instead of ArrayLike hint in call signature
# from numpy.typing import ArrayLike

import itertools
from scipy.special import expit
from scipy.stats import norm
import bisect


class Response(ABC):
    r"""Abstract base class for response function mapping
    :py:class:`TreeNode` objects to ``float`` values given parameters."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __call__(self, node: "TreeNode") -> float:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in self.__dict__.items())})"


class PhenotypeResponse(Response):
    r"""Abstract base class for response function mapping from a
    :py:class:`TreeNode` object's phenotype attribute :math:`x\in\mathbb{R}` to real values given parameters.

    .. math::
        f: \mathbb{R} \to \mathbb{R}

    """

    def __call__(self, node: "TreeNode") -> float:
        return self.f(node.x)

    @abstractmethod
    def f(self, x) -> float:
        r"""Convenience method for computing :math:`f(x)` (e.g. for plotting).

        Args:
            x (array-like): Phenotype value.
        """
        pass


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


class ExponentialResponse(PhenotypeResponse):
    r"""Exponential response function on a :py:class:`TreeNode` object's
    phenotype attribute :math:`x`.

    .. math::
        f(x) = \theta_1 e^{\theta_2 (x - \theta_3)} + \theta_4

    Args:
        xscale: :math:`\theta_2`
        xshift: :math:`\theta_3`
        yscale: :math:`\theta_1`
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


class SigmoidResponse(PhenotypeResponse):
    r"""Sigmoid response function on a :py:class:`TreeNode` object's phenotype
    attribute :math:`x`.

    .. math::
        f(x) = \frac{\theta_1}{1 + e^{-\theta_2 (x - \theta_3)}} + \theta_4

    Args:
        xscale: :math:`\theta_2`
        xshift: :math:`\theta_3`
        yscale: :math:`\theta_1`
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

    def __call__(self, node: "TreeNode") -> float:
        return self.f(node.x)

    def f(self, x) -> float:
        r"""Convenience method for computing :math:`f(x)` (e.g. for plotting).

        Args:
            x (array-like): Phenotype value.
        """
        return self.yscale * expit(self.xscale * (x - self.xshift)) + self.yshift


class Mutator(ABC):
    r"""Abstract base class for generating mutation effects given
    :py:class:`TreeNode` object, which is modified in place."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def mutate(
        self, node: "TreeNode", seed: Optional[Union[int, np.random.Generator]] = None
    ) -> None:
        r"""Mutate a :py:class:`TreeNode` object in place.

        Args:
            node: A :py:class:`TreeNode` to mutate.
            seed: A seed to initialize the random number generation. If ``None``, then fresh,
                  unpredictable entropy will be pulled from the OS.
                  If an ``int``, then it will be used to derive the initial state.
                  If a :py:class:`numpy.random.Generator`, then that will be used directly.
        """
        pass

    @abstractmethod
    def logprob(self, node1: "TreeNode", node2: "TreeNode") -> float:
        r"""Compute the log probability that a mutation effect on ``node1``
        gives ``node2``.

        Args:
            node1: Initial node.
            node2: Mutant node.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in self.__dict__.items())})"


class GaussianMutator(Mutator):
    r"""Gaussian mutation effect on phenotype attribute :math:`x`.

    Args:
        shift: Mean shift wrt current phenotype.
        scale: Standard deviation of mutation effect.
    """

    def __init__(self, shift: float = 0.0, scale: float = 1.0):
        self._distribution = norm(loc=shift, scale=scale)

    def mutate(
        self, node: "TreeNode", seed: Optional[Union[int, np.random.Generator]] = None
    ) -> None:
        node.x += self._distribution.rvs(random_state=seed)

    def logprob(self, node1: "TreeNode", node2: "TreeNode") -> float:
        return self._distribution.logpdf(node2.x - node1.x)


class TreeNode(ete3.Tree):
    r"""A tree generated by a BDMS process. Subclasses
    :py:class:`ete3.TreeNode`.

    Args:
        t: Time of this node.
        x: Phenotype of this node.
        kwargs: Keyword arguments passed to :py:class:`ete3.TreeNode` initializer.
    """

    _BIRTH_EVENT = "birth"
    _DEATH_EVENT = "death"
    _MUTATION_EVENT = "mutation"
    _SURVIVAL_EVENT = "survival"
    _SAMPLING_EVENT = "sampling"

    _OFFSPRING_NUMBER = 2

    _name_generator = itertools.count()

    _time_face = ete3.AttrFace(
        "dist", fsize=6, ftype="Arial", fgcolor="black", formatter="%0.3g"
    )
    _mutation_face = ete3.AttrFace(
        "n_mutations", fsize=6, ftype="Arial", fgcolor="green"
    )

    def __init__(
        self,
        t: float = 0,
        x: float = 0,
        **kwargs: Any,
    ) -> None:
        if "dist" not in kwargs:
            kwargs["dist"] = 0
        if "name" not in kwargs:
            TreeNode._name_generator = itertools.count()
            kwargs["name"] = next(self._name_generator)
        super().__init__(**kwargs)
        self.t = t
        """Time of the node."""
        self.x = x
        """Phenotype of the node."""
        self.event = None
        """Event at this node."""
        self.n_mutations = 0
        """Number of mutations on the branch above this node (zero unless the tree has been pruned above this node,
        removing mutation event nodes)."""
        self._sampled = False
        self._pruned = False

    def evolve(
        self,
        t: float,
        birth_rate: Response = ConstantResponse(1),
        death_rate: Response = ConstantResponse(0),
        mutation_rate: Response = ConstantResponse(1),
        mutator: Mutator = GaussianMutator(shift=0, scale=1),
        birth_mutations: bool = False,
        min_survivors: int = 1,
        retry: int = 1000,
        max_leaves: int = 1000,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        r"""Evolve for time :math:`\Delta t`.

        Args:
            t: Evolve for this time
            birth_rate: Birth rate response function.
            death_rate: Death rate response function.
            mutation_rate: Mutation rate response function.
            mutator: Generator of mutation effects at mutation events
                     (and on offspring of birth events if ``birth_mutations=True``).
            birth_mutations: Flag to indicate whether mutations should occur at birth.
            min_survivors: Minimum number of survivors.
            retry: Number of times to retry if tree goes extinct.
                   A ``RuntimeError`` is raised if it is exceeded during simulation.
            max_leaves: Maximum number of active leaves, to truncate exploding processes.
                        A ``RuntimeError`` is raised if this is exceeded during simulation.
            seed: A seed to initialize the random number generation.
                  If ``None``, then fresh, unpredictable entropy will be pulled from the OS.
                  If an ``int``, then it will be used to derive the initial state.
                  If a :py:class:`numpy.random.Generator`, then that will be used directly.
        """
        if not self.is_root():
            raise TreeError("Cannot evolve a non-root node")
        if self.children:
            raise TreeError(
                f"tree has already evolved at node {self.name} with {len(self.children)} descendant lineages"
            )
        rng = np.random.default_rng(seed)

        success = False
        attempt = 0
        while not success:
            unfinished_nodes = [self]
            # NOTE: this key management is needed because the bisect.insort function does not have a
            #       key argument in python 3.9 (it gains one in 3.10)
            time_keys = [self.t]
            while unfinished_nodes:
                Δt = t - unfinished_nodes[0].t
                new_event_node = unfinished_nodes[0]._generate_event(
                    Δt,
                    birth_rate,
                    death_rate,
                    mutation_rate,
                    mutator,
                    birth_mutations,
                    rng,
                )
                # We pop the corresponding element of unfinished_nodes after we're finished operating on it with
                # _generate_event. The special case is that birth nodes must be operated on twice before they
                # should be popped, if birth_mutations is False. We then insert newly created nodes that need to
                # be operated on. If birth_mutations is True, we get a cherry returned from _generate_event,
                # and need to insert both children.
                if (
                    unfinished_nodes[0].event != self._BIRTH_EVENT
                    or len(unfinished_nodes[0].children) == self._OFFSPRING_NUMBER
                ):
                    unfinished_nodes.pop(0)
                    time_keys.pop(0)
                if new_event_node.event not in (
                    self._DEATH_EVENT,
                    self._SURVIVAL_EVENT,
                ):
                    if new_event_node.event == self._BIRTH_EVENT and birth_mutations:
                        assert len(new_event_node.children) == self._OFFSPRING_NUMBER
                        nodes_to_insert = new_event_node.children
                    else:
                        nodes_to_insert = [new_event_node]
                    for node_to_insert in nodes_to_insert:
                        idx = bisect.bisect(time_keys, node_to_insert.t)
                        time_keys.insert(idx, node_to_insert.t)
                        unfinished_nodes.insert(idx, node_to_insert)
                if len(unfinished_nodes) > max_leaves:
                    raise RuntimeError(
                        f"maximum number of leaves {max_leaves} exceeded during simulation"
                    )

            attempt += 1
            if attempt == retry:
                raise RuntimeError(
                    f"less than {min_survivors} survivors in all {retry} tries"
                )
            if (
                sum(leaf.event == self._SURVIVAL_EVENT for leaf in self)
                >= min_survivors
            ):
                success = True
            else:
                for child in self.children:
                    self.remove_child(child)
                    child.delete()
                TreeNode._name_generator = itertools.count(start=self.name + 1)

    def _generate_event(
        self,
        Δt: float,
        birth_rate: Response,
        death_rate: Response,
        mutation_rate: Response,
        mutator: Mutator,
        birth_mutations: bool,
        rng: np.random.Generator,
    ) -> "TreeNode":
        r"""Simulate a single event, adding a new child TreeNode to self, and
        returning it.

        The Δt parameter is a boundary time: events that reach this time
        are survivors.
        """
        λ = birth_rate(self)
        μ = death_rate(self)
        γ = mutation_rate(self)

        Λ = λ + μ + γ

        τ = min(rng.exponential(1 / Λ), Δt)

        child = TreeNode(
            t=self.t + τ, x=self.x, dist=τ, name=next(self._name_generator)
        )
        if τ == Δt:
            child.event = self._SURVIVAL_EVENT
        else:
            possible_events = [
                self._BIRTH_EVENT,
                self._DEATH_EVENT,
                self._MUTATION_EVENT,
            ]
            event_probabilities = np.array([λ, μ, γ]) / Λ
            child.event = rng.choice(possible_events, p=event_probabilities)
            if child.event == self._BIRTH_EVENT:
                if birth_mutations:
                    for _ in range(self._OFFSPRING_NUMBER):
                        grandchild = TreeNode(
                            t=child.t,
                            x=child.x,
                            dist=0,
                            name=next(self._name_generator),
                        )
                        grandchild.event = self._MUTATION_EVENT
                        mutator.mutate(grandchild, seed=rng)
                        child.add_child(grandchild)
            elif child.event == self._DEATH_EVENT:
                pass
            elif child.event == self._MUTATION_EVENT:
                mutator.mutate(child, seed=rng)
            else:
                raise ValueError(f"unknown event {child.event}")
        return self.add_child(child)

    def sample_survivors(
        self,
        n: Optional[int] = None,
        p: Optional[float] = 1.0,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Choose :math:`n` survivor leaves from the tree, or each survivor leaf with probability
        :math:`p`, to mark as sampled (via the event attribute).

        Args:
            n: Number of leaves to sample.
            p: Probability of sampling a leaf.
            seed: A seed to initialize the random number generation.
                  If ``None``, then fresh, unpredictable entropy will be pulled from the OS.
                  If an ``int``, then it will be used to derive the initial state.
                  If a :py:class:`numpy.random.Generator`, then that will be used directly.
        """
        if self._sampled:
            raise ValueError(f"tree has already been sampled below node {self.name}")
        rng = np.random.default_rng(seed)
        surviving_leaves = [leaf for leaf in self if leaf.event == self._SURVIVAL_EVENT]
        if n is not None:
            for leaf in rng.choice(surviving_leaves, size=n, replace=False):
                leaf.event = self._SAMPLING_EVENT
        elif p is not None:
            for leaf in surviving_leaves:
                if rng.choice(range(2), p=(1 - p, p)):
                    leaf.event = self._SAMPLING_EVENT
        else:
            raise ValueError("must specify either n or p")
        for node in self.traverse():
            node._sampled = True

    def prune(self) -> None:
        r"""Prune the tree to the subtree induced by the sampled leaves.

        Also removes mutation events, and annotates mutation counts in
        child node ``n_mutations`` attribute.
        """
        if self._pruned:
            raise ValueError(f"tree has already been pruned below node {self.name}")
        if not self._sampled:
            raise ValueError(f"tree has not been sampled below node {self.name}")

        event_cache = self.get_cached_content(store_attr="event")
        if self._SAMPLING_EVENT not in event_cache[self]:
            raise TreeError("cannot prune because no leaves were sampled")

        def is_leaf_fn(node):
            return self._SAMPLING_EVENT not in event_cache[node]

        for node in self.iter_leaves(is_leaf_fn=is_leaf_fn):
            parent = node.up
            parent.remove_child(node)
            assert parent.event == self._BIRTH_EVENT
            parent.delete(prevent_nondicotomic=False, preserve_branch_length=True)
        for node in self.traverse(strategy="postorder"):
            if node.event == self._MUTATION_EVENT:
                assert len(node.children) == 1
                node.children[0].n_mutations += 1
                node.delete(prevent_nondicotomic=False, preserve_branch_length=True)
        for node in self.traverse():
            node._pruned = True

    def render(self, *args: Any, cbar_file: Optional[str] = None, **kwargs: Any) -> Any:
        r"""A thin wrapper around :py:func:`ete3.TreeNode.render` that adds some
        custom decoration and a color bar. As with the base class method, pass
        ``"%%inline"`` for the first argument to render inline in a notebook.
        See also ETE's tree rendering `tutorial`_ and linked API docs pages
        there.

        .. _tutorial: http://etetoolkit.org/docs/latest/tutorial/tutorial_drawing.html

        If tree is not pruned, then branches
        are colored according to phenotype, extinct lineages are indicated as
        dotted branches, unsampled non-extint lineages are indicated as solid
        branches, and sampled lineages are indicated as thick solid branches
        with. Sampled leaves are indicated with a circle.

        If tree is pruned, then nodes are colored according to phenotype,
        branches are annotated above with branch length (in black text) and below with number of mutations (in green text).

        Args:
            args: Arguments to pass to :py:func:`ete3.TreeNode.render`.
            cbar_file: If not ``None``, save color bar to this file.
            kwargs: Keyword arguments to pass to :py:func:`ete3.TreeNode.render`.
        """
        if "tree_style" not in kwargs:
            kwargs["tree_style"] = ete3.TreeStyle()
            kwargs["tree_style"].show_leaf_name = False
            kwargs["tree_style"].show_scale = False
        cmap = "coolwarm_r"
        cmap = mpl.cm.get_cmap(cmap)
        halfrange = max(abs(node.x - self.x) for node in self.traverse())
        norm = mpl.colors.CenteredNorm(
            vcenter=self.x,
            halfrange=halfrange if halfrange > 0 else 1,
        )
        colormap = {
            node.name: mpl.colors.to_hex(cmap(norm(node.x))) for node in self.traverse()
        }
        event_cache = self.get_cached_content(store_attr="event")
        for node in self.traverse():
            nstyle = ete3.NodeStyle()
            if not self._pruned:
                if (
                    self._SURVIVAL_EVENT not in event_cache[node]
                    and self._SAMPLING_EVENT not in event_cache[node]
                ):
                    nstyle["hz_line_type"] = 1
                    nstyle["vt_line_type"] = 1
                    nstyle["hz_line_width"] = 0
                elif self._SAMPLING_EVENT not in event_cache[node]:
                    nstyle["hz_line_width"] = 0.5
                else:
                    nstyle["hz_line_width"] = 1
                nstyle["hz_line_color"] = colormap[node.name]
                nstyle["fgcolor"] = colormap[node.name]
                nstyle["size"] = 1 if node.event == self._SAMPLING_EVENT else 0
            else:
                nstyle["fgcolor"] = colormap[node.name]
                if not node.is_root() and not getattr(node.faces, "branch-bottom"):
                    node.add_face(self._time_face, 0, position="branch-top")
                    node.add_face(self._mutation_face, 0, position="branch-bottom")
            node.set_style(nstyle)

        fig = plt.figure(figsize=(2, 1))
        cax = fig.add_axes([0, 0, 1, 0.1])
        plt.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
            orientation="horizontal",
            cax=cax,
            label=r"$x$",
        )
        if cbar_file is not None:
            plt.savefig(cbar_file)

        return super().render(*args, **kwargs)

    def log_likelihood(
        self,
        birth_rate: Response,
        death_rate: Response,
        mutation_rate: Response,
        mutator: Mutator,
        sampling_probability: float,
    ) -> float:
        r"""Compute the log-likelihood of a fully observed tree given the
        specified birth, death, mutation, and sampling parameters.

        Args:
            birth_rate: Birth rate response function.
            death_rate: Death rate response function.
            mutation_rate: Mutation rate response function.
            mutator: Generator of mutation effects.
            sampling_probability: Probability of sampling a survivor.
        """
        if self._pruned:
            raise NotImplementedError("tree must be fully observed, not pruned")
        if not self._sampled:
            raise RuntimeError("tree must be sampled")
        result = 0
        for node in self.iter_descendants():
            Δt = node.dist
            λ = birth_rate(node)
            μ = death_rate(node)
            γ = mutation_rate(node)
            if not 0 <= sampling_probability <= 1:
                raise ValueError("sampling_probability must be in [0, 1]")
            ρ = sampling_probability
            Λ = λ + μ + γ
            logΛ = np.log(Λ)
            if node.event in (self._SAMPLING_EVENT, self._SURVIVAL_EVENT):
                # exponential survival function (no event before sampling time), then sampling probability
                result += -Λ * Δt + np.log(
                    ρ if node.event == self._SAMPLING_EVENT else 1 - ρ
                )
            else:
                if self._MUTATION_EVENT and Δt == 0:
                    # mutation in offspring from birth (simulation run with birth_mutations=True)
                    result += mutator.logprob(node, node.up)
                else:
                    # exponential density for event time
                    result += logΛ - Λ * Δt
                    # multinomial event probability
                    if node.event == self._BIRTH_EVENT:
                        result += np.log(λ) - logΛ
                    elif node.event == self._DEATH_EVENT:
                        result += np.log(μ) - logΛ
                    elif node.event == self._MUTATION_EVENT:
                        result += np.log(γ) - logΛ + mutator.logprob(node, node.up)
                    else:
                        raise ValueError(f"unknown event {node.event}")
        return result
