#!/bin/bash

# TEST ./projects/cf-gcdyn.py --actions simu:dl-infer --n-replicates 1 --label test --version v0 --dry --test --n-trials 6 --n-sub-procs 3

bin=./projects/cf-gcdyn.py
iqvsn=v4  # iqtree version [$iqvsn from taraki-gctree-2021-10/run.sh]
common="--dry --n-max-procs 5 --n-sub-procs 100 --tree-inference-method iqtree --base-outdir /fh/fast/matsen_e/dralph/partis/gcdyn --iqtree-version $iqvsn" #/fh/local/dralph/partis/gcdyn"
dlargs="--model-type-list sigmoid:per-bin --dl-extra-args=\"--epochs 100 --non-sigmoid-input\""  #  --non-sigmoid-input-list 0:1
sig_ranges="--xscale-range-list 0.01,2 --xshift-range-list=-0.5,3 --yscale-range-list 0.5,35"
sxtra="$sig_ranges --simu-extra-args=\"--time-to-sampling-range 17 25 --n-seqs-range 60 95 --n-max-procs 20\""
# echo  ./projects/cf-gcdyn.py --actions dl-infer --n-max-procs 5 --n-sub-procs 100 --base-outdir /fh/fast/matsen_e/dralph/partis/gcdyn --label for-data --version v3 --xscale-range-list 0.01,2 --xshift-range-list=-0.5,3 --yscale-range-list 0.5,35 --carry-cap-range-list 500,1250 --init-population-values-list 8,16,32,64,128 --time-to-sampling-range-list 17,25 --n-seqs-range-list 40,90 --n-replicates 1 --n-trials-list 5000:50000 --simu-extra-args=\"--n-max-procs 20 --initial-birth-rate-range 0.1 0.5\" --learning-rate-list 0.001 --epochs-list 100 --batch-size-list 32 --model-type-list sigmoid:per-bin --non-sigmoid-input-list 0:1 --tree-inference-method iqtree --dry
echo $bin --actions simu --label for-data --version v4 --carry-cap-range-list 500,2000 --init-population-values-list 8,16,32,64,128,256 --n-trials-list 5000:100000 $sxtra $dlargs $common
# exit 0
# dlxtra="--dl-model-label-str for-data/v3/n-trials-50000"
# fixed_sxtra="--n-trials-list 120 --simu-extra-args=\"--carry-cap-values 1500 --time-to-sampling-values 20 --xscale-values 1.4 --xshift-values 1.5 --yscale-values 3.3 --init-population-values 128 --n-seqs-range 60 95\""
# # TODO don't repeat --n-max or --n-sub procs
# # echo $bin --actions simu --label data-mimic --version v4 $fixed_sxtra --n-sub-procs 20 $dlargs $dlxtra $common
# echo $bin --actions simu --label data-mimic --version test-v5 --simu-extra-args=\"--carry-cap-values 750 --time-to-sampling-values 20 --xscale-values 1.4 --xshift-values 1.5 --yscale-values 3.3 --init-population-values 8 --n-seqs-range 60 95\" --n-sub-procs 20 $dlargs $dlxtra $common
# # echo $bin --actions simu --label match-data-sumstats --version v14 --simu-extra-args=\"--carry-cap-values 1000 --time-to-sampling-values 20 --xscale-values 1.6 --xshift-values 2 --yscale-values 7 --init-population-values 32 --n-seqs-range 60 95\" --n-sub-procs 20 $dlargs $dlxtra $common
# exit 0

datadir=/fh/fast/matsen_e/data/taraki-gctree-2021-10/iqtree-processed-data/$iqvsn  # actual data
dlabel=data-iqtree-trained-iqtree-data; dvsn=v4
dsl=combo-trees:d15-wt-trees:d20-wt-trees #:w10-wt-trees:d20-LMP2A-trees

# datadir=/fh/fast/matsen_e/dralph/partis/gcdyn/data-mimic/test-v5  # v4
# dlabel=infer-on-data-mimic; dvsn=v7
# dsl=simu
# # xargs="--dl-extra-args=\"--is-simu\""

xargs="--data-dir $datadir --non-sigmoid-input-list 1 --carry-cap-values-list 500:1250 --init-population-values-list 8:32:128 --simu-extra-args=\"--mutability-multiplier 0.5 --n-seqs-range 60 95\""  # NOTE : rather than , compared to those above (and also carry cap are discrete values here)
for mtype in sigmoid per-bin; do  # can't use --model-type-list since we also need the --dl-model-dir to change
    dld=/fh/fast/matsen_e/dralph/partis/gcdyn/for-data/v3/n-trials-50000/model-type-$mtype/non-sigmoid-input-1/dl-infer/iqtree
    echo $bin --actions data:check-dl:replay-plot-ckdl $common --label $dlabel-$mtype --version $dvsn --dl-model-dir $dld --data-samples-list $dsl --model-type $mtype $xargs
done
