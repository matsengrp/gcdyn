r"""Core GC tree class"""

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
        self.tree: ete3.Tree = tree

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
                    parent.delete(
                        prevent_nondicotomic=False, preserve_branch_length=True
                    )
