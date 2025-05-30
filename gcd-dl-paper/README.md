# zenodo archives

Besides this readme and the script `run.sh`, there are four tgz archives here, with contents detailed below.
Plots are combined via html files, which can be viewed by opening with a browser (for some browsers, permissions may need to be adjusted to allow local file viewing).
Clicking on each image takes you to the underlying svg.

`combo-trees.tgz`:

Replay data, formatted for input to inference step.

`for-data-v6.tgz`:

Simulation training samples and training/testing results. 
Simulation is in subdir `n-trials-50000/simu/`, inference result plots are in subdirs `n-trials-50000/model-type-*/dl-infer/iqtree/*.html`.

`data-iqtree-trained-iqtree-data-v8.tgz`:

Data inference results are in subdirs `data-samples-combo-trees/model-type-*/carry-cap-values-*/init-population-values-*/data/plots.html`.
Data-like simulation using these inferred parameters are in subdirs: `data-samples-combo-trees/model-type-sigmoid/carry-cap-values-*/init-population-values-*/check-dl/`
Summary stat plots are in: `data-samples-combo-trees/model-type-sigmoid/carry-cap-values-*/init-population-values-*/replay-plot-ckdl/plots.html`
And inference result plots on these simulation samples is in: `data-samples-combo-trees/model-type-sigmoid/carry-cap-values-*/init-population-values-*/dl-infer/iqtree/plots.html`
The sample with the summary stats that best matched data is: `carry-cap-values-500/init-population-values-128/death-values-0.2`.

# installation

We recommend installing via Docker (quick start guide [here](https://docs.docker.com/get-started/)):

```
sudo docker pull quay.io/matsengrp/gcdyn-dl
sudo docker run -it --name container-1 -v ~:/home/mambauser/host quay.io/matsengrp/gcdyn-dl /bin/bash
```
The `-v` mounts your home directory on the host machine to the path `/home/mambauser/host` inside the container, so we can pass in files from the host machine and easily extract results.
This mounted directory (whether it's your `$HOME` or not) should contain the REPLAYDIR and DATADIR paths (described below) with replay data files.

You can also install [gcdyn](https://matsengrp.github.io/gcdyn/developer.html) and [partis](https://github.com/psathyrella/partis/blob/main/docs/install.md) by hand.

# dowload and running

You'll need to download the replay data by cloning the github repo:

```
git clone git@github.com:matsengrp/gcreplay.git
```
Note that this repo contains all per-sequence info, but you also need the iqtree-inferred trees from zenodo (`combo-trees.tgz`).
Before using the run script `run.sh`, you need to edit the three paths at the top of it (output dir, gcreplay dir, and iqtree-inferred tree dir).
Note that running the entirety of this script at once would be very time consuming (mostly, some of the simulation might take a week).
However, it's easy to pick and choose what you want to run, for instance re-doing network training and inference, or remaking plots, is much faster.

Each of the commands in `run.sh` runs a number of different actions such as making simulation, summary statistic plotting, and network training and inference.
For concision, they are each listed with all necessary actions in one command; however in practice it is usually better to run only a few steps at a time, since some steps are computationally expensive (may take days or longer on smaller machines) and thus may need some shepherding to run to completion.
The different actions are listed under `--actions`, and for the most part must be run in the order specified (however they will generally crash informatively if they don't find their input).
Simulation, for instance, is usually the slowest step, so it may make sense to run it by itself.
Any results that are already completed will be found and skipped before running any new jobs, so there's no harm in telling it to run actions for which either partial or complete results are already present.
Remove the `--dry` when you're ready to actually run (after veryifying that the printed commands look like they're correct).
