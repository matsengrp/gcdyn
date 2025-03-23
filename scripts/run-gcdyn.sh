#!/bin/bash

# zenodo dir: /fh/fast/matsen_e/dralph/partis/gcdyn/zenodo/

# TEST ./projects/cf-gcdyn.py --actions simu:dl-infer --n-replicates 1 --label test --version v0 --dry --test --n-trials 6 --n-sub-procs 3

bin=./projects/cf-gcdyn.py
iqvsn=v4  # iqtree version [$iqvsn from taraki-gctree-2021-10/run.sh]
common="--dry --n-max-procs 5 --n-sub-procs 100 --tree-inference-method iqtree --base-outdir /fh/fast/matsen_e/dralph/partis/gcdyn --iqtree-version $iqvsn" #/fh/local/dralph/partis/gcdyn"
dlargs="--model-type-list sigmoid:per-bin --dl-extra-args=\"--epochs 35\""
sig_ranges="--xscale-range-list 0.01,2 --xshift-range-list=-0.5,3 --yscale-range-list 0.5,35"
sxtra="$sig_ranges --simu-extra-args=\"--time-to-sampling-range 17 25 --n-seqs-range 60 95 --n-max-procs 20\""
# # old: echo  ./projects/cf-gcdyn.py --actions dl-infer --n-max-procs 5 --n-sub-procs 100 --base-outdir /fh/fast/matsen_e/dralph/partis/gcdyn --label for-data --version v3 --xscale-range-list 0.01,2 --xshift-range-list=-0.5,3 --yscale-range-list 0.5,35 --carry-cap-range-list 500,1250 --init-population-values-list 8,16,32,64,128 --time-to-sampling-range-list 17,25 --n-seqs-range-list 40,90 --n-replicates 1 --n-trials-list 5000:50000 --simu-extra-args=\"--n-max-procs 20 --initial-birth-rate-range 0.1 0.5\" --learning-rate-list 0.001 --epochs-list 100 --batch-size-list 32 --model-type-list sigmoid:per-bin --non-sigmoid-input-list 0:1 --tree-inference-method iqtree --dry
# # old: echo $bin --actions simu --label for-data --version v4 --carry-cap-range-list 500,1500 --init-population-values-list 8,16,32,64,128,256 --n-trials-list 5000:100000 $sxtra $dlargs $common
# echo $bin --actions simu --label for-data --version v5 --carry-cap-range-list 500,2000 --init-population-values-list 8,16,32,64,128 --n-trials-list 1000:25000 $sxtra $dlargs $common

dlxtra="--dl-model-label-str for-data/v5/n-trials-25000 --dl-model-vars model-type"
dmargs="--xscale-values-list 1.5 --xshift-values-list 1.9 --simu-extra-args=\"--carry-cap-values 2000 --init-population-values 8 --mutability-multiplier 0.5 --time-to-sampling-values 20 --n-seqs-range 60 95 --n-trials 120\" --n-sub-procs 30"
# echo $bin --actions simu --label data-mimic --version v7-death --yscale-values-list 6.5 --death-values-list 0.05:0.1:0.2:0.5 $dlargs $dlxtra $common $dmargs
# echo $bin --actions simu --label data-mimic --version v7-yscale --yscale-values-list 5:10:20 $dlargs $dlxtra $common $dmargs
# echo $bin --actions simu --label x-ceil --version v5 --yscale-values-list 6.5 --birth-response-list sigmoid-ceil --x-ceil-start-values-list 1.5 $dlargs $dlxtra $common $dmargs
echo $bin --actions simu --label thanasis --version v0 $dlargs $dlxtra $common --simu-extra-args=\"--xscale-values 1.5 --xshift-values -0.1 --yscale-values 2.5 --yshift 0.6 --death-response constant --min-survivors 0 --death-values 1 --carry-cap-values 1000 --init-population-values 2 --mutability-multiplier 1 --time-to-sampling-values 15 --sample-fraction 0.1 --capacity-method hard --n-trials 120\" --n-sub-procs 30
exit 0

datadir=/fh/fast/matsen_e/data/taraki-gctree-2021-10/iqtree-processed-data/$iqvsn  # actual data
dlabel=data-iqtree-trained-iqtree-data; dvsn=v7
dsl=combo-trees:d15-wt-trees:d20-wt-trees #:w10-wt-trees:d20-LMP2A-trees

xargs="--data-dir $datadir --carry-cap-values-list 500:750:1500:2000 --init-population-values-list 8:32:128 --simu-extra-args=\"--mutability-multiplier 0.5 --n-seqs-range 60 95\""  # NOTE : rather than , compared to those above (and also carry cap are discrete values here)
echo $bin --actions data:check-dl:replay-plot-ckdl:dl-infer $common --label $dlabel --version $dvsn --data-samples-list $dsl $xargs $dlargs $dlxtra --check-dl-input
# NOTE i ran the per-bin 'dl-infer' action of this by hand, since i'd need to edit the indir to come from the sigmoid check-dl, which, UGH it's just too much:
#  gcd-dl infer --is-simu --indir /fh/fast/matsen_e/dralph/partis/gcdyn/data-iqtree-trained-iqtree-data/v7/data-samples-combo-trees/model-type-sigmoid/carry-cap-values-2000/init-population-values-8/check-dl/iqtree --outdir /fh/fast/matsen_e/dralph/partis/gcdyn/data-iqtree-trained-iqtree-data/v7/data-samples-combo-trees/model-type-per-bin/carry-cap-values-2000/init-population-values-8/dl-infer/iqtree --epochs 35 --model-dir /fh/fast/matsen_e/dralph/partis/gcdyn/for-data/v5/n-trials-25000/model-type-per-bin/dl-infer/iqtree --model-type per-bin
