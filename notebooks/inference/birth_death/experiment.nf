#!/usr/bin/env -S nextflow run -resume

workflow {
    def experiment_paths = Channel.fromPath("experiments/**/mcmc_config.py") |
        map { it.parent.toString() }
        // Store as string to avoid hashing the entire directory for caching
    
    def simulation_results = experiment_paths |
        map { [it, file(it).parent / "tree_config.py"] } |
        sample_trees |
        map { [it[0], it[1], file(it[0]) / "mcmc_config.py", it[2]] } |
        run_mcmc

    simulation_results | map { it[0..1] } | summarize_trees

    //simulation_results | map { [it[0], file(it[0]).parent.parent / "plots.qmd", it[2]] } | visualize
}

process sample_trees {
    input:
    tuple val(experiment_path), path(tree_config_file)

    output:
    tuple val(experiment_path), path(tree_config_file), path("trees.pkl")
 
    """
    #!/usr/bin/env python

    import pickle

    from ${tree_config_file.baseName} import (
        NUM_TREES,
        PRESENT_TIME,
        INITIAL_STATE,
        TRUE_PARAMETERS,
    )
    from gcdyn.utils import sample_trees

    tree_collections = []

    for tree_seed in range(100):
        trees = sample_trees(
            n=NUM_TREES,
            t=PRESENT_TIME,
            init_x=INITIAL_STATE,
            **TRUE_PARAMETERS,
            seed=tree_seed,
            min_survivors=1,
            prune=True,
        )

        tree_collections.append(trees)

    with open("trees.pkl", "wb") as f:
        pickle.dump(tree_collections, f)
    """
}

process run_mcmc {
    input:
    tuple val(experiment_path), path(tree_config_file), path(mcmc_config_file), path("trees.pkl")

    output:
    tuple val(experiment_path), path("trees.pkl"), path("samples.csv")
 
    """
    #!/usr/bin/env python

    import pickle
    import pandas as pd

    from ${mcmc_config_file.baseName} import (
        MCMC_SEED,
        NUM_MCMC_SAMPLES,
        MCMC_PARAMETERS,
        log_likelihood
    )
    from gcdyn.mcmc import mh_chain
    from functools import partial
    from jax import jit
    from jax.config import config

    config.update("jax_enable_x64", True)

    with open("trees.pkl", "rb") as f:
        tree_collections = pickle.load(f)

    samples = pd.DataFrame()

    for run, trees in enumerate(tree_collections):
        posterior_samples, stats = mh_chain(
            length=NUM_MCMC_SAMPLES,
            parameters=MCMC_PARAMETERS,
            log_likelihood=jit(partial(log_likelihood, trees=trees)),
            seed=MCMC_SEED
        )

        samples = pd.concat((samples, pd.DataFrame(dict(**posterior_samples, **stats, run=run))))

    samples.to_csv("samples.csv")
    """
}

process summarize_trees {
    publishDir "$experiment_path", mode: "copy"

    input:
    tuple val(experiment_path), path("trees.pkl")

    output:
    path "tree_summary.txt"

    """
    #!/usr/bin/env python

    import pickle

    from collections import defaultdict

    with open("trees.pkl", "rb") as f:
        tree_collections = pickle.load(f)

    with open("tree_summary.txt", "w") as f:
        type_counts_nodes = defaultdict(int)
        type_counts_leaves = defaultdict(int)

        for collection in tree_collections:
            for tree in collection:
                for node in tree.traverse():
                    type_counts[node.x] += 1
                    
                    if node.is_leaf():
                        type_counts_leaves[node.x] += 1

        f.write(f"Number of total nodes: {sum(1 for _ in tree.traverse())}\\n")
        for type in sorted(type_counts_nodes.keys()):
            f.write(f"  Type {type} exists in {type_counts[type]} nodes\\n")
        f.write("\\n")
        f.write(f"Number of leaves: {sum(1 for _ in tree.iter_leaves())}\\n")
        for type in sorted(type_counts_leaves.keys()):
            f.write(f"  Type {type} exists in {type_counts[type]} leaves\\n")

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