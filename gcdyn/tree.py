r"""Core GC tree class"""

import jax.numpy as np
from jax import random

from jax.scipy.special import expit
import ete3
import matplotlib as mp
from IPython.display import display

from gcdyn.parameters import Parameters


class Tree:
    r"""A class that represents one complete GC tree

    Args:
        T: simulation sampling time
        seed: random seed
        params: model parameters
    """

    def __init__(self, T: float, seed: int, params: Parameters):
        self.params: Parameters = params
        # store most recently used node name so we can ensure unique node names
        self._name = 0
        # initialize root
        tree = ete3.Tree(name=self._name, dist=0)
        tree.t = 0
        tree.x = 0
        tree.event = None
        self.evolve(tree, T, random.PRNGKey(seed))
        self.tree: ete3.Tree = tree

    def λ(self, x: float):
        r"""Birth rate of phenotype x

        Args:
            x: phenotype

        Returns:
            float: birth rate
        """
        return self.params.θ[0] * expit(self.params.θ[1] * (x - self.params.θ[2]))

    def evolve(self, tree: ete3.Tree, t: float, key: random.PRNGKeyArray):
        r"""Evolve an ETE Tree node with a phenotype attribute for time t

        Args:
            tree:initial tree to evolve
            t: sampling time
            key: random key
        """
        λ_x = self.λ(tree.x)
        Λ = λ_x + self.params.μ + self.params.m
        time_key, event_key = random.split(key)
        τ = random.exponential(time_key) / Λ
        if τ > t:
            self._name += 1
            child = ete3.Tree(name=self._name, dist=t)
            child.x = tree.x
            child.t = tree.t + t
            child.event = (
                "sampled" if random.uniform(event_key) < self.params.ρ else "unsampled"
            )
            tree.add_child(child)
            return

        possible_events = ["birth", "death", "mutation"]
        event_probabilities = np.array([λ_x, self.params.μ, self.params.m]) / Λ
        event = possible_events[
            random.choice(event_key, len(possible_events), p=event_probabilities)
        ]
        self._name += 1
        child = ete3.Tree(name=self._name, dist=τ)
        child.t = tree.t + τ
        child.x = tree.x
        child.event = event
        if event == "birth":
            child1_key, child2_key = random.split(event_key)
            self.evolve(child, t - τ, child1_key)
            self.evolve(child, t - τ, child2_key)
        elif event == "death":
            pass
        elif event == "mutation":
            mutation_key, child_key = random.split(event_key)
            child.x += random.normal(mutation_key)
            self.evolve(child, t - τ, child_key)
        else:
            raise ValueError("unknown event")
        tree.add_child(child)

    def _decorate(self):
        r"""Add node style to the tree

        Leaf node that is sampled is indicated as green
        """
        cmap = "coolwarm_r"
        cmap = mp.cm.get_cmap(cmap)

        # define the minimum and maximum values for our colormap
        normalizer = mp.colors.CenteredNorm(
            vcenter=0, halfrange=max(abs(node.x) for node in self.tree.traverse())
        )
        colormap = {
            node.name: mp.colors.to_hex(cmap(normalizer(node.x)))
            for node in self.tree.traverse()
        }

        for node in self.tree.traverse():
            nstyle = ete3.NodeStyle()
            nstyle["hz_line_color"] = colormap[node.name]
            nstyle["hz_line_width"] = 2

            if node.is_root() or node.event in set(["birth", "death", "mutation"]):
                nstyle["size"] = 0
            elif node.event == "sampled":
                nstyle["fgcolor"] = "green"
                nstyle["size"] = 5
            elif node.event == "unsampled":
                nstyle["fgcolor"] = "grey"
                nstyle["size"] = 5
            else:
                raise ValueError(f"unknown event {node.event}")

            node.set_style(nstyle)

    def draw_tree(self, output_file: str = None):
        r"""Visualizes the tree

        If output file is given, tree visualization is saved to the file.
        If not, tree visualization is rendered to the notebook.

        Args:
            output_file: name of the output file of the tree visualization. Defaults to None.
        """
        self._decorate()
        ts = ete3.TreeStyle()
        ts.scale = 100
        ts.branch_vertical_margin = 3
        ts.show_leaf_name = True
        ts.show_scale = False
        if output_file is None:
            display(self.tree.render("%%inline", tree_style=ts))
        else:
            self.tree.render(output_file, tree_style=ts)

    def prune(self):
        """Prune the tree to the subtree induced by the sampled leaves."""
        event_cache = self.tree.get_cached_content(store_attr="event")
        for node in self.tree.iter_descendants(
            is_leaf_fn=lambda node: "sampled" not in event_cache[node]
        ):
            if "sampled" not in event_cache[node]:
                parent = node.up
                parent.remove_child(node)
                if parent.event == "birth":
                    parent.children[0].dist += parent.dist
                    parent.delete(
                        prevent_nondicotomic=False, preserve_branch_length=False
                    )
