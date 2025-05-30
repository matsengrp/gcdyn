#!/bin/bash
# started from https://github.com/matsengrp/gcdyn/blob/ab2d637e4ccdeccb48f9f5037ca64766eb77b3c4/scripts/run-gcdyn.sh

outdir=/fh/fast/matsen_e/dralph/partis/gcdyn  # change this to the desired output dir
replaydir=/fh/fast/matsen_e/shared/replay/gcreplay  # change this to the location to which you cloned the gcreplay repository from https://github.com/matsengrp/gcreplay (used for per-sequence meta info, e.g. nextflow/results/merged-results/gctree-node-data.csv)
datadir=/fh/fast/matsen_e/data/taraki-gctree-2021-10/iqtree-processed-data/v4  # change this to the location to which you unpacked the iqtree-inferred replay data trees from zenodo (combo-trees.tgz)

# first run this command to open a docker container
image_name=matsengrp/gcdyn  # gcd-local
echo sudo docker run \
     --mount type=bind,src=$outdir,dst=/gcdyn-dl-output \
     --mount type=bind,ro,src=$replaydir,dst=/gcreplay \
     --mount type=bind,ro,src=$datadir,dst=/data-trees \
     --user root --rm -it --name container-1 $image_name /bin/bash
# exit 0

# these three dirs are correct if using the docker run command above, but otherwise will need modification
export outdir=/gcdyn-dl-output
export replaydir=/gcreplay
export datadir=/data-trees

# then choose one of the following commands to run inside that container

bin=./projects/cf-gcdyn.py
common="--dry --n-max-procs 5 --n-sub-procs 100 --tree-inference-method iqtree --base-outdir $outdir --gcreplay-dir $replaydir"
dlargs="--model-type-list sigmoid:per-bin --dl-extra-args=\"--epochs 35\""
sig_ranges="--xscale-range-list 0.01,2 --xshift-range-list=-0.5,3 --yscale-range-list 0.5,35 --yshift-range-list 0,0.6 --death-range-list 0.05,0.5"
sxtra="$sig_ranges --simu-extra-args=\"--time-to-sampling-range 10 35 --n-seqs-range 50 130 --n-max-procs 20\""
echo $bin --actions simu:replay-plot:dl-infer --label for-data --version v6 --carry-cap-range-list 500,2000 --init-population-values-list 8,16,32,64,128 --n-trials-list 1000:50000 $sxtra $dlargs $common

dlabel=data-iqtree-trained-iqtree-data; dvsn=v8
dsl=combo-trees:d15-wt-trees:d20-wt-trees

xargs="--data-dir $datadir --carry-cap-values-list 500:750:1500:2000 --init-population-values-list 8:32:128 --simu-extra-args=\"--mutability-multiplier 0.5 --n-seqs-range 60 95\""  # NOTE : rather than , compared to those above (and also carry cap are discrete values here)
echo $bin --actions data:check-dl:replay-plot-ckdl:dl-infer $common --label $dlabel --version $dvsn --data-samples-list $dsl $xargs $dlargs $dlxtra --check-dl-input
