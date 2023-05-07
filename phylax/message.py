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
with an *update operator* :math:`F_{\uparrow}: \mathbb{R}^d\times\mathbb{R}^m \to \mathbb{R}^r` that
combines the local feature vector with an incoming message :math:`m_v \in \mathbb{R}^m`.

.. math::

    h_v = F_{\uparrow}\left(x_v, m_v\right).

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

Similarly, we can define a pre-order traversal scheme, which simplifies because tree nodes have at
most one parent. The message received by node :math:`v` is a vector :math:`m_v \in \mathbb{R}^m`
computed via an aggregate operator
:math:`G_{\downarrow}: \left(\mathbb{R}^r \cup \emptyset\right)\times\mathbb{R}^{d} \to \mathbb{R}^m`
over the representations of the parent nodes of :math:`v`, denoted
:math:`\mathcal{P}_v \subset \mathcal{V}`.
Note :math:`|\mathcal{P}_v| \in \{0, 1\}` for trees.

.. math::

    m_v = G_{\downarrow}\left(\left\{h_u : u \in \mathcal{P_v}\right\}, x_v\right).


Implementation
--------------

To facilitate compatibility with :py:func:`jax.jit`, the implementation has
the following properties:

- Operations are handled via a functional interface (i.e. without stateful updates).
- Although the model specification lends itself to recursion, the termination
  conditions would depend on tree topology, which is not defined at trace time. We
  therefore avoid tree recursion.
- Trees are represented with sparse arrays, and message propagation is handled via
  array scans with carried state. This ensures that all operations can be traced, and
  that new tree topologies can be handled without recompilation.


References
----------

.. [1] Veronika Thost and Jie Chen. "Directed Acyclic Graph Neural Networks."
       *International Conference on Learning Representations*. 2021.
       https://openreview.net/forum?id=JbuYF437WB6
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import sparse
import equinox as eqx
import abc
import ete3
import numpy as onp

from typing import Optional, Tuple, Sequence
from jaxtyping import Array, Bool, Int, Float


class Messenger(eqx.Module, abc.ABC):
    r"""Abstract base classs for functions that aggregate representations from neighbor nodes
    in a permutation-invariant (set-pooling) manner, and combine them with the focal node's
    features to produce a message.

    Concrete subclasses must define ``null_value`` as an :py:class:`eqx.Module` field.
    """
    null_value: Float

    @abc.abstractmethod
    def __call__(
        self,
        neighbor_representations: Float[Array, "n r"],
        focal_node_features: Float[Array, " d"],
    ) -> Float[Array, " m"]:
        r"""Aggregate neighbor representations and combine with focal node features to produce a message.

        Args:
            neighbor_representations: :math:`n\times r` matrix of neighbor node representations, where
                                      :math:`n` is the number of neighbor nodes and :math:`r` is the
                                      representation dimension.
            focal_node_features: The focal node's feature :math:`d`-vector

        Returns:
            message: Message :math:`m`-vector.
        """


class Updater(eqx.Module, abc.ABC):
    r"""Abstract base classs for functions that combine a node's feature vector with an incoming message to produce
    an updated representation.

    Concrete subclasses must define output representation dimension :math:`r` as an :py:class:`eqx.Module` field.
    """
    r: Int

    @abc.abstractmethod
    def __call__(
        self, node_features: Float[Array, " d"], message: Float[Array, " m"]
    ) -> Float[Array, " r"]:
        r"""Combine a node's feature vector with an incoming message to produce an updated representation.

        Args:
            node_features: The node's feature :math:`d`-vector
            message: The incoming message :math:`m`-vector

        Returns:
            updated_representation: The node's updated representation :math:`r`-vector.
        """


class TreeMessagePasser(eqx.Module):
    r"""Messages can be passed from the leaves to the root, or from the root to the leaves,
    and can be aggregated in user-defined ways as functions of :py:class:`ete3.TreeNode` attributes.

    Args:
        tree: The tree to pass messages through.
        up_messenger: A function that aggregates representations of child nodes and a parent node's features to a
                      message. It should be invariant wrt child node permutations.
        down_messenger: A function that aggregates representations of parent nodes and a child node's features to a
                        message. It should be invariant wrt parent node permutations.
        updater: A function that combines the feature vector of a node with an incoming message and returns the
                 node's updated representation.
        downdater: A function that combines the feature vector of a node with an incoming message and returns the
                   node's updated representation.
    """
    up_messenger: Messenger
    down_messenger: Messenger
    updater: Updater
    downdater: Updater
    pre_order_idxs: Int[Array, " n"]
    post_order_idxs: Int[Array, " n"]
    parents: Bool[Array, "n n"]
    children: Bool[Array, "n n"]
    branch_lengths: Float[Array, " n"]
    leaves: Bool[Array, " n"]

    def __init__(
        self,
        tree: ete3.Tree,
        up_messenger: Optional[Messenger] = None,
        down_messenger: Optional[Messenger] = None,
        updater: Optional[Updater] = None,
        downdater: Optional[Updater] = None,
    ) -> None:
        pre_order_idx_map = {
            node: idx for idx, node in enumerate(tree.traverse(strategy="preorder"))
        }
        post_order_idx_map = {
            node: idx for idx, node in enumerate(tree.traverse(strategy="postorder"))
        }
        assert pre_order_idx_map.keys() == post_order_idx_map.keys()

        self.pre_order_idxs = jnp.array(
            [pre_order_idx_map[node] for node in pre_order_idx_map]
        )
        self.post_order_idxs = jnp.array(
            [pre_order_idx_map[node] for node in post_order_idx_map]
        )
        self.branch_lengths = jnp.array([node.dist for node in pre_order_idx_map])

        n = len(self.pre_order_idxs)
        assert self.pre_order_idxs[0] == 0
        assert self.post_order_idxs[-1] == 0
        assert pre_order_idx_map[tree] == 0
        assert post_order_idx_map[tree] == n - 1

        self.leaves = jnp.zeros(n, dtype=bool)
        parent_idxs = []
        child_idxs = []
        for node, idx in pre_order_idx_map.items():
            if node.is_leaf():
                self.leaves = self.leaves.at[idx].set(True)
            if not node.is_root():
                parent_idxs.append([idx, pre_order_idx_map[node.up]])
            if not node.is_leaf():
                for child in node.children:
                    child_idxs.append([idx, pre_order_idx_map[child]])

        assert sum(self.leaves) == len(tree)
        assert len(parent_idxs) == len(child_idxs) == n - 1

        # form sparse matrices
        # NOTE: 1/0 sparse to bool seems necessary because sparse matrices don't
        #       fully support bool dtype
        self.parents = sparse.BCOO(([1] * len(parent_idxs), parent_idxs), shape=(n, n))
        self.children = sparse.BCOO(([1] * len(child_idxs), child_idxs), shape=(n, n))

        self.up_messenger = up_messenger
        self.down_messenger = down_messenger
        self.updater = updater
        self.downdater = downdater

    def initialize_features(
        self,
        tree: ete3.Tree,
        attributes: Sequence[str],
    ) -> Float[Array, "n d"]:
        r"""Initialize :math:`n\times d` node feature matrix with rows in pre-order.

        Args:
            tree: The tree to initialize messages for.
            attributes: Names of node attributes to use as features. The columns of the
                        feature matrix will be ordered according to this sequence.
        """
        return jnp.asarray(
            [
                [getattr(node, attribute) for attribute in attributes]
                for node in tree.traverse(strategy="preorder")
            ]
        )

    def decorate(
        self, node_features: Float[Array, "n d"], tree: ete3.Tree, attribute: str
    ) -> None:
        r"""Add node features to tree as node attributes.

        .. warning::

            The feature vector is cast to a numpy array.

        Args:
            node_features: Array of node representations in pre-order.
            tree: The tree to decorate.
            attribute: Name of node attribute to decorate with.
        """
        for idx, node in enumerate(tree.traverse(strategy="preorder")):
            node.add_feature(attribute, onp.asarray(node_features[idx]))

    @eqx.filter_jit
    def _update(
        self, carry: Tuple[Float[Array, "n r"], Float[Array, "n d"]], row_idx: Int
    ) -> Tuple[Tuple[Float[Array, "n r"], Float[Array, "n d"]], Float[Array, "n r"]]:
        r"""Update function for :py:func:`jax.lax.scan` to the representation of one node.

        Args:
            carry: Carried tuple of ``(partially_updated_representations, node_features)``.
            row_idx: The focal node's row index.
        """
        partially_updated_representations = carry[0]
        node_features = carry[1]
        # NOTE: transforming to dense and casting 1/0 sparse to bool seems necessary
        #       because sparse matrices don't fully support bool dtype
        mask = self.children[row_idx].todense().astype(bool)[:, None]
        masked_representations = jnp.where(
            mask, partially_updated_representations, self.up_messenger.null_value
        )
        message = self.up_messenger(masked_representations, node_features[row_idx])
        updated_representations = partially_updated_representations.at[row_idx].set(
            self.updater(node_features[row_idx], message)
        )
        return (updated_representations, node_features), updated_representations

    @eqx.filter_jit
    def upward(
        self, features: Float[Array, "n d"]
    ) -> Tuple[Float[Array, "n r"], Float[Array, "n n r"]]:
        r"""Generate node representations by propagating messages from leaves to root.

        Args:
            features: A pre-ordered :math:`n\times d` array of node features.

        Returns:
            representations: A pre-ordered :math:`n\times r` array of node representations.
        """
        initial_representations = jnp.full(
            (features.shape[0], self.updater.r), self.up_messenger.null_value
        )
        carry = (initial_representations, features)
        (representations, features), representations_trajectory = jax.lax.scan(
            self._update, carry, self.post_order_idxs
        )
        return representations, representations_trajectory

    @eqx.filter_jit
    def _downdate(
        self, carry: Tuple[Float[Array, "n r"], Float[Array, "n d"]], row_idx: Int
    ) -> Tuple[Tuple[Float[Array, "n r"], Float[Array, "n d"]], Float[Array, "n r"]]:
        r"""Downdate function for :py:func:`jax.lax.scan` to the representation of one node.

        Args:
            carry: Carried tuple of ``(partially_downdated_representations, node_features)``.
            row_idx: The focal node's row index.
        """
        partially_downdated_representations = carry[0]
        node_features = carry[1]
        # NOTE: transforming to dense and casting 1/0 sparse to bool seems necessary
        #       because sparse matrices don't fully support bool dtype
        mask = self.parents[row_idx].todense().astype(bool)[:, None]
        masked_representations = jnp.where(
            mask, partially_downdated_representations, self.down_messenger.null_value
        )
        message = self.down_messenger(masked_representations, node_features[row_idx])
        downdated_representations = partially_downdated_representations.at[row_idx].set(
            self.downdater(node_features[row_idx], message)
        )
        return (downdated_representations, node_features), downdated_representations

    @eqx.filter_jit
    def downward(
        self, features: Float[Array, "n d"]
    ) -> Tuple[Float[Array, "n r"], Float[Array, "n n r"]]:
        r"""Generate node representations by propagating messages from root to leaves.

        Args:
            features: A pre-ordered :math:`n\times d` array of node features.

        Returns:
            representations: A pre-ordered :math:`n\times r` array of node representations.
        """
        initial_representations = jnp.full(
            (features.shape[0], self.downdater.r), self.down_messenger.null_value
        )
        carry = (initial_representations, features)
        (representations, features), representations_trajectory = jax.lax.scan(
            self._downdate, carry, self.pre_order_idxs
        )
        return representations, representations_trajectory


# NOTE: for jit compilations that persist across trees, we want to pad the various matrices

# NOTE: filter_jit missing the integer indexing

# NOTE: consider adding a batch 1st dimension, to handle many trees. It might be possible to make this vmap compatible...

# NOTE: TOPOLOGICAL BATCHING as in Thost et al. (2020) https://arxiv.org/pdf/2101.07965.pdf, we could send subtrees to the GPU
