r"""Message passing on trees.

This module provides a JAX-compatible message-passing framework for ETE3 trees.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import ete3
import abc

from typing import Optional, Iterable, Tuple
from collections.abc import Callable, Sequence
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, PyTree


class TreeMessagePasser(eqx.Module):
    r"""Tree message passer. Messages can be passed from the leaves to the root, or from the root to the leaves,
    and can be aggregated in user-defined ways as functions of :py:class:`ete3.TreeNode` attributes.
    The framework is designed to be compatible with :py:func:`jax.jit` and :py:func:`jax.grad`.

    Args:
        tree: The tree to pass messages through.
        attributes: Collection of node attribute names to use as features.
        aggregator: A function that aggregates messages from children to parent nodes.
        updater: A function that updates messages at a node.
        downdater: A function that updates messages at a node.

    Example:

        Import the necessary modules:
        >>> import ete3
        >>> from gcdyn.messages import TreeMessagePasser
        >>> tree = ete3.Tree("((A:1,B:1):1,C:2);")
        >>> attributes = ("dist",)
        >>> aggregator = lambda x: jnp.sum(x, axis=0)
        >>> updater = lambda x, y: x + y
        >>> downdater = lambda x, y: x + y
        >>> message_passer = TreeMessagePasser(tree, attributes, aggregator, updater, downdater)
        >>> # message_passer.messages
    """
    tree: ete3.TreeNode
    attributes: Sequence[str]
    aggregator: Callable[[Float[Array, "p m"]], Float[Array, " m"]]
    updater: Callable[[Float[Array, " d"], Float[Array, " m"]], Float[Array, " m"]]
    downdater: Callable[[Float[Array, " d"], Float[Array, " m"]], Float[Array, " m"]]
    features: Float[Array, "n d"]
    parent_indicators: Bool[Array, "n n"]
    child_indicators: Bool[Array, "n n"]
    messages: Float[Array, "n m"]  # NOTE: mutatable state!!!!!!!!!!!!!

    def __init__(
        self,
        tree: ete3.TreeNode,
        attributes: Sequence[str],
        aggregator: eqx.Module,
        updater: Optional[eqx.Module] = None,
        downdater: Optional[eqx.Module] = None,
    ) -> None:
        self.tree = tree
        self.attributes = attributes
        post_order_idxs = {
            node: idx
            for idx, node in enumerate(self.tree.traverse(strategy="postorder"))
        }
        n = len(post_order_idxs)
        d = len(self.attributes)
        # m = ???
        leaf_indicator = jnp.zeros(n, dtype=bool)
        self.features = jnp.empty((n, d))
        self.parent_indicators = jnp.zeros((n, n), dtype=bool)
        self.child_indicators = jnp.zeros((n, n), dtype=bool)
        for node, idx in post_order_idxs.items():
            self.features = self.features.at[idx].set(
                jnp.array([getattr(node, attribute) for attribute in self.attributes])
            )
            if not node.is_root():
                self.parent_indicators = self.parent_indicators.at[
                    idx, post_order_idxs[node.up]
                ].set(True)
            if not node.is_leaf():
                self.child_indicators = self.child_indicators.at[
                    idx, [post_order_idxs[child] for child in node.children]
                ].set(True)

        self.messages = jnp.zeros((n, 10))

        self.aggregator = aggregator
        self.updater = updater
        self.downdater = downdater

    def upward(self) -> None:
        r"""Pass messages from leaves to root."""
        for node in self.tree.traverse("postorder"):
            self.updater(node)

    def downward(self) -> None:
        r"""Pass messages from root to leaves."""
        for node in self.tree.traverse("preorder"):
            self.downdater(node)



if __name__ == "__main__":
    import doctest
    doctest.testmod()


# NOTE: eqx.filter_grad and eqx.filter_jit automatically pick out jax.array pytrees as dynamic arguments, and the
#       rest as static arguments. Careful, edges may be int, not float.

# NOTE: we may need to manage state, i.e. the messages. see https://docs.kidger.site/equinox/api/nn/stateful/

# NOTE: for jit compilations that persist across trees, maybe we want to pad the various matrices
