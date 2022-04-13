import jax.numpy as np
from jax import random
from jax.scipy.stats import norm
from jax import jit, grad
from jax.scipy.special import expit
import ete3
import matplotlib as mp
import matplotlib.pyplot as plt
# import mushi.optimization as opt


class GC_tree:

    def __init__(self, T, key, θ, μ, m, ρ):
        key, _ = random.split(key)
        self.θ = θ
        self.μ = μ
        self.m = m
        self.ρ = ρ
        
        while True:
            # initialize root
            tree = ete3.Tree(name=0, dist=0)
            tree.add_feature("t", 0)
            tree.add_feature("x", 0)
            tree.add_feature("event", None)
            # get new seed
            key, _ = random.split(key)
            self.evolve(tree, T, key)
            if 50 < len(tree) < 96:
                print(f"size {len(tree)}")
                break
        self.tree = tree

    def λ(self, x):
        r"""Birth rate of phenotype x"""
        return self.θ[0] * expit(self.θ[1] * (x - self.θ[2]))

    def evolve(self, tree, t, key):
        r"""Evolve an ETE Tree node with a phenotype attribute for time t"""
        λ_x = self.λ(tree.x)
        Λ = λ_x + self.μ + self.m
        time_key, event_key = random.split(key)
        τ = random.exponential(time_key) / Λ
        if τ > t:
            child = ete3.Tree(name=tree.name + 1, dist=t)
            child.add_feature("x", tree.x)
            child.add_feature("t", tree.t + t)
            child.add_feature("event", "sampled" if random.uniform(event_key) < self.ρ else "unsampled")
            tree.add_child(child)
            return

        possible_events = ["birth", "death", "mutation"]
        event_probabilities = np.array([λ_x, self.μ, self.m]) / Λ
        event = possible_events[random.choice(event_key, len(possible_events), p=event_probabilities)]

        child = ete3.Tree(name=tree.name + 1, dist=τ)
        child.add_feature("t", tree.t + τ)
        child.add_feature("x", tree.x)
        child.add_feature("event", event)
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
            raise ValueError(f"unknown event")
        tree.add_child(child)

    def draw_tree(self, file_name):
        cmap = "coolwarm_r"
        cmap = mp.cm.get_cmap(cmap)

        # define the minimum and maximum values for our colormap
        # normalizer = mp.colors.Normalize(vmin=-10, vmax=10)
        normalizer = mp.colors.CenteredNorm(vcenter=0,
                                            halfrange=max(abs(node.x) for node in self.tree.traverse()))
        colormap = {node.name: mp.colors.to_hex(cmap(normalizer(node.x)))
                    for node in self.tree.traverse()}

        ts = ete3.TreeStyle()
        ts.scale = 100
        ts.branch_vertical_margin = 3
        ts.show_leaf_name = False
        ts.show_scale = False

        for node in self.tree.traverse():
            nstyle = ete3.NodeStyle()
            nstyle["hz_line_color"] = colormap[node.name]
            nstyle["hz_line_width"] = 2
            if node.is_leaf() and node.event == "sampled":
                nstyle["fgcolor"] = "green"
                nstyle["size"] = 3
            else:
                nstyle["size"] = 0
            node.set_style(nstyle)

        self.tree.render(file_name, tree_style=ts)
    
    def log_likelihood(self):
        result = 0
        for node in self.tree.children[0].traverse():
            x = node.up.x
            Δt = node.dist
            λ_x = self.λ(x)
            Λ = λ_x + self.μ + self.m
            logΛ = np.log(Λ)
            if node.event in ("sampled", "unsampled"):
                # exponential survival function (no event before sampling time), then sampling probability
                result += - Λ * Δt + np.log(self.ρ if node.event == "sampled" else 1 - self.ρ)
            else:
                # exponential density for event time
                result += logΛ - Λ * Δt
                # multinomial event probability
                if node.event == "birth":
                    result += np.log(λ_x) - logΛ
                elif node.event == "death":
                    result += np.log(self.μ) - logΛ
                elif node.event == "mutation":
                    Δx = node.x - x
                    result += np.log(self.m) - logΛ + norm.logpdf(Δx)
                else:
                    raise ValueError(f"unknown event {node.event}")
        return result
