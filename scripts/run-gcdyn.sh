#!/bin/bash

# zenodo dir: /fh/fast/matsen_e/dralph/partis/gcdyn/zenodo/
# see also readme and scripts in gcdyn/gcd-dl-paper/

# TEST ./projects/cf-gcdyn.py --actions simu:dl-infer --n-replicates 1 --label test --version v0 --dry --test --n-trials 6 --n-sub-procs 3

bin=./projects/cf-gcdyn.py
iqvsn=v4  # iqtree version [$iqvsn from taraki-gctree-2021-10/run.sh]
common="--dry --n-max-procs 5 --n-sub-procs 100 --tree-inference-method iqtree --base-outdir /fh/fast/matsen_e/dralph/partis/gcdyn --iqtree-version $iqvsn" #/fh/local/dralph/partis/gcdyn"
dlargs="--model-type-list sigmoid:per-bin" # --epochs-list 35:100 --prebundle-layer-cfg-list big:huge --dropout-rate-list 0:0.3" #--dl-extra-args=\"--epochs 35\""
# ----------------------------------------------------------------------------------------
# v6 (current paper version):
# sig_ranges="--xscale-range-list 0.01,2 --xshift-range-list=-0.5,3 --yscale-range-list 0.5,35 --yshift-range-list 0,0.6 --death-range-list 0.05,0.5"
# sxtra="$sig_ranges --simu-extra-args=\"--time-to-sampling-range 10 35 --n-seqs-range 50 130 --n-max-procs 20\""
# echo $bin --actions simu --label for-data --version v6 --carry-cap-range-list 500,2000 --init-population-values-list 8,16,32,64,128 --n-trials-list 1000:50000 $sxtra $dlargs $common
# ----------------------------------------------------------------------------------------
# v7 (somewhat narrower ranges, maybe infer with this after submitting):
sig_ranges="--xscale-range-list 0.75,2 --xshift-range-list=1,2.5 --yscale-range-list 2.5,35 --yshift-range-list 0,0.75 --death-range-list 0.05,0.75"
sxtra="$sig_ranges --simu-extra-args=\"--time-to-sampling-range 17 25 --n-seqs-range 60 95 --n-max-procs 20\""
echo $bin --actions simu --label for-data --version v7 --carry-cap-range-list 500,2000 --init-population-values-list 8,16,32,64,128 --n-trials-list 1000:50000 $sxtra $dlargs $common
# # sig_ranges="--xscale-range-list 1.5,2 --xshift-range-list=0.6,0.9 --yscale-range-list 8,12 --yshift-range-list 0.4,0.6 --death-range-list 0.2,0.5"
# # sxtra="$sig_ranges --simu-extra-args=\"--time-to-sampling-range 30 35 --n-seqs-range 60 95 --n-max-procs 20\""
# # echo $bin --actions simu --label sig-ceil-train --version v0 --carry-cap-range-list 500,2000 --init-population-values-list 8,16,32,64,128 --n-trials-list 1000:10000 $sxtra $dlargs $common
# exit 0

dlxtra="--dl-model-label-str for-data/v6/n-trials-50000 --dl-model-vars model-type:epochs:prebundle-layer-cfg:dropout-rate" #model-type"
dmargs="--xscale-values-list 1 --xshift-values-list 1.75 --yshift-values-list 0.3 --simu-extra-args=\"--carry-cap-values 1250 --init-population-values 32 --death-values 0.2 --mutability-multiplier 0.5 --time-to-sampling-values 20 --n-seqs-range 60 95 --n-trials 120\" --n-sub-procs 30"
echo $bin --actions simu --label central-train --version v1 --yscale-values-list 15 $dlargs $dlxtra $common $dmargs  # NOTE don't really need this, dl-infer action on check-dl below does ~same thing, and we use that for the paper
# NOTE v9 here is from the overtrained sample, but I'm not actually using these in the paper atm so don't need to figure out which previous version I would want to use (I think it'd be v8)
# echo $bin --actions simu --label data-mimic --version v9 --yscale-values-list 13.2 $dlargs $dlxtra $common $dmargs  # NOTE don't really need this, dl-infer action on check-dl below does ~same thing, and we use that for the paper
# echo $bin --actions simu --label data-mimic --version v9-yscale --yscale-values-list 5:10:15:20:30 $dlargs $dlxtra $common $dmargs
# echo $bin --actions simu --label x-ceil --version v7 --yscale-values-list 13.2 --birth-response-list sigmoid-ceil --x-ceil-start-values-list 1.5 $dlargs $dlxtra $common $dmargs
# # echo $bin --actions simu --label thanasi --version v1 $dlargs $dlxtra $common --yshift-values-list 0:0.1:0.6 --death-values-list 0.1:0.5 --capacity-method-list birth:hard --simu-extra-args=\"--xscale-values 1.5 --xshift-values -0.1 --yscale-values 2.5 --death-response constant --min-survivors 100 --carry-cap-values 1000 --init-population-values 2 --mutability-multiplier 1 --time-to-sampling-values 15 --sample-fraction 0.1 --n-trials 120\" --n-sub-procs 30
# # echo $bin --actions simu --label sig-ceil --version v0 $dlargs $dlxtra $common --simu-extra-args=\"--xscale-values 2.2 --xshift-values 0.8 --yscale-values 10 --yshift-values 0.5 --death-values 0.4 --carry-cap-values 1200 --init-population-values 64 --time-to-sampling-values 33 --n-seqs-range 120 121 --n-trials 120\" --n-sub-procs 30
# # echo $bin --actions simu --label sig-ceil --version v1 $dlargs $dlxtra $common --simu-extra-args=\"--xscale-values 1.2 --xshift-values 0.8 --yscale-values 10 --yshift-values 0.5 --death-values 0.4 --carry-cap-values 1200 --init-population-values 64 --time-to-sampling-values 33 --n-seqs-range 120 121 --n-trials 120\" --n-sub-procs 30
# # echo $bin --actions simu --label sig-ceil --version test-v2 $dlargs $dlxtra $common --yscale-values-list 2:5:10 --simu-extra-args=\"--xscale-values 1.5 --xshift-values -0.1 --yshift-values 0.6 --death-values 0.5 --carry-cap-values 1000 --init-population-values 2 --time-to-sampling-values 30 --n-seqs-range 120 121 --n-trials 30\" --n-sub-procs 30
exit 0

datadir=/fh/fast/matsen_e/data/taraki-gctree-2021-10/iqtree-processed-data/$iqvsn  # actual data
dlabel=data-iqtree-trained-iqtree-data; dvsn=v8  # v9 was the overtrained results
dsl=combo-trees:d15-wt-trees:d20-wt-trees #:w10-wt-trees:d20-LMP2A-trees

# xargs="--data-dir $datadir --carry-cap-values-list 500:750:1000:2000 --init-population-values-list 8:32:64:128 --death-values-list 0.05:0.1:0.2:0.4 --simu-extra-args=\"--mutability-multiplier 0.5 --n-seqs-range 60 95\""  # NOTE : rather than , compared to those above (and also carry cap are discrete values here)
xargs="--data-dir $datadir --carry-cap-values-list 500:1000 --init-population-values-list 32:128 --death-values-list 0.4:0.5 --simu-extra-args=\"--mutability-multiplier 0.5 --n-seqs-range 60 95\""  # NOTE : rather than , compared to those above (and also carry cap are discrete values here)
echo $bin --actions data:check-dl:replay-plot-ckdl:dl-infer $common --label $dlabel --version $dvsn --data-samples-list $dsl $xargs $dlargs $dlxtra --check-dl-input
# NOTE i ran the per-bin 'dl-infer' action of this by hand, since i'd need to edit the indir to come from the sigmoid check-dl, which, UGH it's just too much:
#  gcd-dl infer --is-simu --indir /fh/fast/matsen_e/dralph/partis/gcdyn/data-iqtree-trained-iqtree-data/v8/data-samples-combo-trees/model-type-sigmoid/carry-cap-values-500/init-population-values-128/death-values-0.2/check-dl/iqtree --outdir /fh/fast/matsen_e/dralph/partis/gcdyn/data-iqtree-trained-iqtree-data/v8/data-samples-combo-trees/model-type-per-bin/carry-cap-values-500/init-population-values-128/death-values-0.2/dl-infer/iqtree --model-dir /fh/fast/matsen_e/dralph/partis/gcdyn/for-data/v6/n-trials-50000/model-type-per-bin/dl-infer/iqtree --model-type per-bin


# param-degen: copied some input/output files and edited them by hand, then ran this (also modifed svgs by hand to remove title, extra legend, and righthand y axis):
# gcd-dl infer --is-simu --indir /fh/fast/matsen_e/dralph/partis/gcdyn/param-degen-v0 --outdir $fs/partis/gcdyn/param-degen-v0/inferred-output --model-dir /fh/fast/matsen_e/dralph/partis/gcdyn/for-data/v6/n-trials-50000/model-type-sigmoid/dl-infer/iqtree --model-type sigmoid
