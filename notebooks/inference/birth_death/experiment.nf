#!/usr/bin/env -S nextflow run -resume

workflow {
    def experiment_paths = Channel.fromPath("experiments/**/config.py") |
        map { it.parent.toString() }
        // Store as string to avoid hashing the entire directory for caching
    
    def simulation_results = experiment_paths |
        map { [ it, file(it) / "config.py"] } |
        simulate

    simulation_results | map { it[0..2] } | summarize_trees

    simulation_results | map { [it[0], file(it[0]) / "plots.qmd", it[3]] } | visualize
}

process simulate {
    input:
    tuple val(experiment_path), path(config_file)

    output:
    tuple val(experiment_path), path(config_file), path("trees.pkl"), path("samples.csv")
 
    """
    #!/usr/bin/env python

    import pickle
    import pandas as pd
    
    from ${config_file.baseName} import (
        NUM_TREES,
        PRESENT_TIME,
        INITIAL_STATE,
        TRUE_PARAMETERS,
        TREE_SEED,
        MCMC_PARAMETERS,
        log_likelihood
    )
    from gcdyn.utils import sample_trees
    from gcdyn.mcmc import mh_chain
    from functools import partial
    from jax import jit
    from jax.config import config

    trees = sample_trees(
        n=NUM_TREES,
        t=PRESENT_TIME,
        init_x=INITIAL_STATE,
        **TRUE_PARAMETERS,
        seed=TREE_SEED,
        min_survivors=1,
        prune=True,
    )

    with open("trees.pkl", "wb") as f:
        pickle.dump(trees, f)

    config.update("jax_enable_x64", True)

    posterior_samples, stats = mh_chain(
        length=2000,
        parameters=MCMC_PARAMETERS,
        log_likelihood=jit(partial(log_likelihood, trees=trees)),
    )

    pd.DataFrame(dict(**posterior_samples, **stats)).to_csv("samples.csv")
    """
}

process summarize_trees {
    publishDir "$experiment_path", mode: "copy"

    input:
    tuple val(experiment_path), path(config_file), path("trees.pkl")

    output:
    path "tree_summary.txt"

    """
    #!/usr/bin/env python

    import pickle

    from collections import defaultdict

    with open("trees.pkl", "rb") as f:
        trees = pickle.load(f)

    with open("tree_summary.txt", "w") as f:
        for i, tree in enumerate(trees):
            f.write(f"Tree {i}\\n")
            f.write(f"  Number of total nodes: {sum(1 for _ in tree.traverse())}\\n")
            

            type_counts = defaultdict(int)

            for node in tree.traverse():
                type_counts[node.x] += 1

            for type in sorted(type_counts.keys()):
                f.write(f"  Type {type} exists in {type_counts[type]} nodes\\n")

            f.write("\\n")

            f.write(f"  Number of leaves: {sum(1 for _ in tree.iter_leaves())}\\n")

            type_counts = defaultdict(int)

            for node in tree.iter_leaves():
                type_counts[node.x] += 1

            for type in sorted(type_counts.keys()):
                f.write(f"  Type {type} exists in {type_counts[type]} leaves\\n")
            
            f.write("\\n")
    """
}

process visualize {
    publishDir "$experiment_path", mode: "copy"

    input:
    tuple val(experiment_path), path(quarto_file), path(samples_csv)

    output:
    path "plots.pdf"
 
    """
    quarto render $quarto_file \
        --output plots.pdf \
        --execute-dir \$(pwd)
    """
}