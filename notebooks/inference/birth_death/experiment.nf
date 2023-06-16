// To run experiments: nextflow run experiment.nf -resume

workflow {
    def simulation_scripts = Channel.fromPath("experiments/**/simulation.py")
    def experiment_dirs = simulation_scripts.map {file(it).parent}
    def analysis_scripts = experiment_dirs.map {it / "analysis.qmd"}

    simulate(simulation_scripts)
    analyze(analysis_scripts, simulate.out, experiment_dirs)
}

process simulate {
    input:
    path experiment_script

    output:
    path "samples.csv"
 
    """
    python $experiment_script
    """
}

process analyze {
    publishDir "$output_dir", mode: "copy"

    input:
    path analysis_script
    path samples_file
    val output_dir

    output:
    path "analysis.pdf"
 
    """
    quarto render $analysis_script \
        --output analysis.pdf \
        --execute-dir \$(pwd)
    """
}