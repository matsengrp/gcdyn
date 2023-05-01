r"""Message passing on rooted trees
===================================

This module provides a JAX-compatible message-passing framework for ETE3 trees.

Mathematical form of models
---------------------------

.. note::

    We adapt notation from Thost and Chen (2021) [1]_.

We have a rooted tree :math:`\mathcal{T} = (\mathcal{V}, \mathcal{E})` with
:math:`n = |\mathcal{V}|` nodes and :math:`n-1` edges, where :math:`\mathcal{V} = \{1, \ldots, n\}`
is the set of nodes and :math:`\mathcal{E} \subset \mathcal{V} \times \mathcal{V}` is the set of edges.

Each node :math:`v \in \mathcal{V}` has an input feature vector :math:`x_v \in \mathbb{R}^d`.
We will compute its representation :math:`h_v \in \mathbb{R}^r` via a message-passing scheme
with an *update operator* :math:`F: \mathbb{R}^d\times\mathbb{R}^m \to \mathbb{R}^r` that combines
the local feature vector with an incoming message :math:`m_v \in \mathbb{R}^m`.

.. math::

    h_v = F\left(x_v, m_v\right).

Post-order message passing
^^^^^^^^^^^^^^^^^^^^^^^^^^

Using a post-order traversal over nodes, the message received by node :math:`v` is a vector
:math:`m_v \in \mathbb{R}^m` computed via an *aggregate operator*
:math:`G_{\uparrow}: 2^{\mathbb{R}^r}\times\mathbb{R}^{d} \to \mathbb{R}^m` over the representations
of the child nodes of :math:`v`, denoted :math:`\mathcal{C}_v \subset \mathcal{V}`.

.. math::

    m_v = G_{\uparrow}\left(\left\{h_u : u \in \mathcal{C_v}\right\}, x_v\right).

The dependence on the *set* of child representations implies that the aggregate operator is
*permutation-equivariant* wrt child node ordering.

Pre-order message passing
^^^^^^^^^^^^^^^^^^^^^^^^^

Similarly, we can define a pre-order traversal scheme, which simplifies because the aggregate
operator :math:`G_{\downarrow}:\mathbb{R}^r\times\mathbb{R}^{d} \to \mathbb{R}^m` takes exactly
one parent node representation, rather than a set of child nodes.

.. math::

    m_v = G_{\downarrow}\left(h_{p(v)}, x_v\right).

where :math:`p:\mathcal{V}\to\mathcal{V}` is the parent function.


Implementation
--------------

To facilitate compatibility with :py:func:`jax.jit`, the implementation has
the following properties:

- Operations are handled via a functional interface (i.e. without stateful updates).
- Although the model specification lends itself to recursion, the termination
  conditions would depend on tree topology, which is not defined at trace time. We
  therefore avoid tree recursion.
- Trees are represented with sparse matrices, and message propagation is handled via
  scans over matrix indices. This ensures that all operations can be traced, and that
  new tree topologies can be handled without recompilation.


References
----------

.. [1] Veronika Thost and Jie Chen. "Directed Acyclic Graph Neural Networks."
       *International Conference on Learning Representations*. 2021.
       https://openreview.net/forum?id=JbuYF437WB6


Examples
--------

>>> import ete3
>>> from phylax.treemp import TreeMessagePasser

"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import sparse
import equinox as eqx
import ete3
import numpy as onp

from typing import Optional, Tuple
from collections.abc import Callable, Sequence
from jaxtyping import Array, Bool, Int, Float


class Aggregator(eqx.Module):
    r"""Aggregates messages in a node-equivariant (set-pooling) manner.

    Args:
        init: Initial fill value for aggregated message (must be consistent with ``operation``).
        operation: The operation
    """
    init: float
    operation: Callable[[Float[Array, " m"], Float[Array, " m"]], Float[Array, " m"]]

    def __init__(
        self,
        init: float = 0.0,
        operation: Callable[[Float[Array, " p m"]], Float[Array, " m"]] = jax.lax.add,
    ) -> None:
        super().__init__(init, operation)

    def __call__(self, messages: Float[Array, " p m"]) -> Float[Array, " m"]:
        r"""Aggregate messages.

        Args:
            messages: :math:`p\times m` matrix of messages, where :math:`p` is the number of
                    messages and :math:`m` is the message dimension.
        """
        return super().__call__(messages)


class TreeMessagePasser(eqx.Module):
    r"""Messages can be passed from the leaves to the root, or from the root to the leaves,
    and can be aggregated in user-defined ways as functions of :py:class:`ete3.TreeNode` attributes.

    Args:
        tree: The tree to pass messages through.
        attributes: Collection of node attribute names to use as features.
        up_messenger: A function that aggregates representations of child nodes and a parent node's features to a
                      message. It should be invariant wrt child node permutations.
        down_messenger: A function that aggregates a parent node's representation and a child node's features to a
                        message.
        updater: A function that combines the feature vector of a node with an incoming message and returns the
                 node's updated representation.
        downdater: A function that combines the feature vector of a node with an incoming message and returns the
                   node's updated representation.
        mask_value: Value to use for masked representations. Must be consistent with the aggregation in
                    :attr:`up_messenger`. The default value of 0.0 is appropriate for any additive aggregation.

    Example:

        Import the necessary modules:

        >>> import ete3
        >>> from phylax.treemp import TreeMessagePasser

        Define a tree:

        >>> tree = ete3.Tree("((A:1,B:1):1,C:2);")

        Names of tree node attributes to use as features:

        >>> attributes = ("dist",)

        Define the messenger and updater functions:

        >>> messenger = lambda x, y: jnp.sum(x, axis=0)
        >>> updater = lambda x, y: x + y

        Initialize the message passer:

        >>> message_passer = TreeMessagePasser(tree, attributes, up_messenger=messenger, updater=updater)
    """
    up_messenger: Callable[
        [Float[Array, "c r"], Float[Array, " d"]], Float[Array, " m"]
    ]
    down_messenger: Callable[
        [Float[Array, " r"], Float[Array, " d"]], Float[Array, " m"]
    ]
    updater: Callable[[Float[Array, " d"], Float[Array, " m"]], Float[Array, " r"]]
    downdater: Callable[[Float[Array, " d"], Float[Array, " m"]], Float[Array, " r"]]
    leaves: Bool[Array, " n"]
    features: Float[Array, "n d"]
    parents: Bool[Array, "n n"]
    children: Bool[Array, "n n"]
    mask_value: Float

    def __init__(
        self,
        tree: ete3.Tree,
        attributes: Sequence[str],
        up_messenger: Optional[eqx.Module] = None,
        down_messenger: Optional[eqx.Module] = None,
        updater: Optional[eqx.Module] = None,
        downdater: Optional[eqx.Module] = None,
        mask_value: Float = 0.0,
    ) -> None:
        pre_order_idxs = {
            node: idx for idx, node in enumerate(tree.traverse(strategy="preorder"))
        }
        post_order_idxs = {
            node: idx for idx, node in enumerate(tree.traverse(strategy="postorder"))
        }
        assert pre_order_idxs.keys() == post_order_idxs.keys()

        n = len(post_order_idxs)
        d = len(attributes)

        self.features = jnp.empty((n, d))

        # Due to post-order indexing, the upper triangle of the adjacency matrix
        # denotes parents, and the lower triangle denotes children. We will use
        # sparse matrices to store these relationships, with a parent (child)
        # matrix containing the upper (lower) triangle of the adjacency matrix.
        # We'll also store a sparse array of leaf node indicators.

        leaf_idxs = []
        parent_idxs = []
        child_idxs = []
        for node, idx in post_order_idxs.items():
            if node.is_leaf():
                leaf_idxs.append([idx])
            self.features = self.features.at[idx].set(
                jnp.array([getattr(node, attribute) for attribute in attributes])
            )
            if not node.is_root():
                parent_idxs.append([idx, post_order_idxs[node.up]])
            if not node.is_leaf():
                for child in node.children:
                    child_idxs.append([idx, post_order_idxs[child]])

        assert len(leaf_idxs) == len(tree)
        assert len(parent_idxs) == len(child_idxs) == n - 1

        # form sparse matrices
        # NOTE: 1/0 sparse to bool seems necessary because sparse matrices don't
        #       fully support bool dtype
        self.leaves = sparse.BCOO(([1] * len(leaf_idxs), leaf_idxs), shape=(n,))
        self.parents = sparse.BCOO(([1] * len(parent_idxs), parent_idxs), shape=(n, n))
        self.children = sparse.BCOO(([1] * len(child_idxs), child_idxs), shape=(n, n))

        self.up_messenger = up_messenger
        self.down_messenger = down_messenger
        self.updater = updater
        self.downdater = downdater
        self.mask_value = mask_value

    def initialize_representations(
        self,
        tree: ete3.Tree,
        node_initializer: Callable[[ete3.Tree], Float[Array, " r"]],
    ) -> Float[Array, "n r"]:
        r"""Initialize :math:`n\times r` node representation matrix with rows in post-order.

        Args:
            tree: The tree to initialize messages for.
            node_initializer: A function that initializes the representation
                              :math:`r`-vector for a given node.
        """
        return jnp.asarray(
            [node_initializer(node) for node in tree.traverse(strategy="postorder")]
        )

    def decorate(
        self, representations: Float[Array, "n r"], tree: ete3.Tree, attribute: str
    ) -> None:
        r"""Add representation to tree as node attributes.

        .. warning::

            The representation is cast to a numpy array.

        Args:
            representations: Array of node representations.
            tree: The tree to decorate.
            attribute: Name of node attribute to decorate with.
        """
        for idx, node in enumerate(tree.traverse(strategy="postorder")):
            node.add_feature(attribute, onp.asarray(representations[idx]))

    @jax.jit
    def _up_message(
        self,
        representations: Float[Array, "n r"],
        node_idx: Int,
    ) -> Float[Array, " m"]:
        r"""Aggregate representations from the children of a focal node, and the
        focal node's features, into a message.

        Args:
            representations: Array of node representations.
            node_idx: Post-order index of the focal node.
        """
        # NOTE: transforming to dense and casting 1/0 sparse to bool seems necessary
        #       because sparse matrices don't fully support bool dtype
        mask = self.children[node_idx].todense().astype(bool)[:, None]
        masked_representations = jnp.where(mask, representations, self.mask_value)
        return self.up_messenger(masked_representations, self.features[node_idx])

    @jax.jit
    def _update(
        self, representations: Float[Array, "n r"], node_idx: Int
    ) -> Tuple[Float[Array, "n r"], Float[Array, "n r"]]:
        r"""Update the representations of one node.

        Args:
            representations: Array of node representations.
            node_idx: Post-order index of the node to update.

        Returns:
            (updated_representations, updated_representations): Updated array of node representations,
                                                                duplicated as a tuple for compatibility
                                                                with :py:func:`jax.lax.scan`.
        """
        message = self._up_message(representations, node_idx)
        return (
            representations.at[node_idx].set(
                self.updater(self.features[node_idx], message)
            ),
        ) * 2

    @jax.jit
    def upward(
        self, representations: Float[Array, "n r"]
    ) -> Tuple[Float[Array, "n r"], Float[Array, "n n r"]]:
        r"""Update node representations by propagating messages from leaves to root.

        Args:
            representations: An :math:`n\times r` array of node representations.

        Returns:
            final_representations: An :math:`n\times r` array of final node representations.
            representations_trajectory: An :math:`n\times n\times r` array of node representations
                                        at each step of the upward pass.
        """

        def f(representations, node_idx):
            r"""This function simply duplicates the output of :py:meth:`_update`, as needed for :py:func:`jax.lax.scan`."""
            return (self._update(representations, node_idx),) * 2

        return jax.lax.scan(
            lambda representations, node_idx: (self._update(representations, node_idx),)
            * 2,
            representations,
            jnp.arange(representations.shape[0]),
        )

    def downward(self, representations: Float[Array, "n r"]) -> Float[Array, "n r"]:
        r"""Update representations from root to leaves."""
        # for node in self.tree.traverse("preorder"):
        #     self.downdater(node)


# NOTE: can I implement this with recursion? Is it then a recursive neural network? This might be a bad idea
#       because the stopping condition depends on the tree structure, so is not traceable by JAX

# NOTE: aggregation needs to be a node-equivariant (set pooling) operation

# NOTE: eqx.filter_grad and eqx.filter_jit automatically pick out jax.array pytrees as dynamic arguments, and the
#       rest as static arguments. Careful, edges may be int, not float.

# NOTE: we may need to manage state, i.e. the messages. see https://docs.kidger.site/equinox/api/nn/stateful/

# NOTE: for jit compilations that persist across trees, maybe we want to pad the various matrices

# NOTE: consider adding a batch 1st dimension, to handle many trees. It might be possible to make this vmap compatible...

# NOTE: TOPOLOGICAL BATCHING as in Thost et al. (2020) https://arxiv.org/pdf/2101.07965.pdf, we could send subtrees to the GPU

# Potential names:
#  - phylox (see https://www.youtube.com/watch?v=D8hSU223yYI)
#  - xylem (see https://en.wikipedia.org/wiki/Xylem)
