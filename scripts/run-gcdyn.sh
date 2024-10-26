#!/bin/bash

# TEST ./projects/cf-gcdyn.py --actions simu:dl-infer --n-replicates 1 --label test --version v0 --dry --test --n-trials 6 --n-sub-procs 3

bin=./projects/cf-gcdyn.py
common="--actions simu --dry --n-max-procs 5 --n-sub-procs 100 --base-outdir  /fh/fast/matsen_e/dralph/partis/gcdyn" #/fh/local/dralph/partis/gcdyn"
# echo ./projects/cf-gcdyn.py --actions simu --version v1 --birth-response-list constant:soft-relu:soft-relu:soft-relu --xscale-list 1:1:2:3 --zip-vars birth-response:xscale $commone
xscales=0.1:1:5:10; xshifts=-5:-2:-1:0:1:2:5
# common="--actions simu --final-plot-xvar xshift --dry --n-replicates 2"
# echo $bin $common --version v2 --birth-response-list soft-relu --xscale-list $xscales --xshift-list=$xshifts
# echo $bin $common --version v3-carry-cap --birth-response-list sigmoid --xscale-list $xscales --xshift-list=$xshifts
# echo $bin $common --version v4-carry-cap --birth-response-list sigmoid --xscale-list 0.5:0.75:1:1.25:1.5 --xshift-list=0:1:2:5:8:15
# common="--actions simu:merge-simu:process --final-plot-xvar xshift --dry"
# echo $bin $common --version v6 --birth-response-list sigmoid --xscale-list 0.75:1:1.25 --xshift-list=2:5 --n-replicates 20
# echo $bin $common --label vs-n-trees --version v0 --birth-response-list sigmoid --carry-cap-list 150 --xscale-list 0.5,1.5 --xshift-list 2 --n-replicates 2 --n-trials-list 100:1000 --perf-metrics xscale --plot-metrics dl-infer
# echo $bin $common --label vs-n-trees --version v1 --birth-response-list sigmoid --carry-cap-list 150 --xscale-list 1 --xshift-list 2 --n-replicates 2 --n-trials-list 100:500 --perf-metrics xscale --plot-metrics dl-infer
# echo $bin $common --label vs-n-leaves --version v2 --carry-cap-list 150 --xscale-list 0.5,0.75,1,1.25,1.5 --xshift-list 2 --n-replicates 2 --n-seqs-list 50:75:100:125 --n-trials-list 100:1000 --perf-metrics all-dl --model-size-list small:tiny --plot-metrics dl-infer --final-plot-xvar n-seqs --pvks-to-plot 1000
# echo $bin $common --label vs-n-trees --version v6 --carry-cap-list 150 --xscale-list 0.5,1,1.5,2,2.5,3,3.5,4,4.5,5 --xshift-list=-0.5,0,0.5,1,1.5,2,2.5,3 --n-replicates 2 --n-trials-list 500:5000:50000:500000 --test-xscale-values-list 1.5,2,2.5,3,3.5,4 --test-xshift-values-list 0.5,1,1.5,2,2.5 --simu-extra-args=\"--n-max-procs 20\" --perf-metrics all-test-dl --plot-metrics group-expts --final-plot-xvar n-trees-per-expt --n-trees-per-expt-list 1:5:10:50:100
# echo $bin $common --label vs-n-trees --version v7 --carry-cap-list 150 --xscale-list 2 --xshift-list=-0.5,3 --n-replicates 2 --n-trials-list 500:5000:50000 --simu-extra-args=\"--n-max-procs 20\" --perf-metrics all-test-dl --plot-metrics group-expts --final-plot-xvar n-trees-per-expt --n-trees-per-expt-list 1:5:10:50:100
# echo ./projects/cf-gcdyn.py --actions dl-infer --n-max-procs 2 --base-outdir /fh/fast/matsen_e/dralph/partis/gcdyn --label test-simu-test --version v1 --carry-cap-list 150 --xscale-values-list 2 --xshift-range-list=-0.5,3 --n-replicates 2 --n-trials-list 500:5000 --simu-extra-args="--n-max-procs 20 --n-trees-per-param-set 10" --dl-extra-args="--no-shuffle" --perf-metrics all-test-dl --plot-metrics group-expts --final-plot-xvar n-trees-per-expt --n-trees-per-expt-list 1:5 --params-to-predict xshift --dry
# echo ./projects/cf-gcdyn.py --actions dl-infer --n-max-procs 4 --base-outdir /fh/fast/matsen_e/dralph/partis/gcdyn --label vary-trees-per-param-set --version v0 --carry-cap-list 150 --xscale-values-list 2 --xshift-range-list=-0.5,3 --n-replicates 2 --n-trials-list 500:5000 --n-trees-per-param-set-list 10:50 --simu-extra-args="--n-max-procs 20" --dl-extra-args="--no-shuffle" --perf-metrics all-test-dl --plot-metrics group-expts --final-plot-xvar n-trees-per-expt --n-trees-per-expt-list 1:5 --params-to-predict xshift --dry
# echo $bin $common --label bundle-xshift --version v8 --carry-cap-list 150 --xscale-values-list 2 --xshift-range-list=-0.5,3 --n-replicates 1 --n-trials-list 5000:50000 --simu-bundle-size 50 --simu-extra-args=\"--n-max-procs 20\" --perf-metrics all-test-dl --plot-metrics group-expts --final-plot-xvar n-trees-per-expt --n-trees-per-expt-list 1:5 --params-to-predict xshift
# NOTE in line below, --simu-bundle-size is 250 for 500k sample (could use zip vars for this, but also want to zip n-trials with epochs)
# echo $bin $common --label bundle-xscale-xshift --version v0 --carry-cap-list 150 --xscale-range-list 0.5,5 --xshift-range-list=-0.5,3 --n-replicates 1 --n-trials-list 5000:50000:500000 --epochs-list 1000:5000:1000 --zip-vars n-trials:epochs --simu-bundle-size 50 --simu-extra-args=\"--n-max-procs 20\" --params-to-predict xscale:xshift --dropout-rate-list 0:0.2:0.5 --learning-rate-list 0.001:0.01 --ema-momentum-list 0.9:0.99
# pranges="--xscale-range-list 0.5,5 --xshift-range-list=-0.5,3 --yscale-range-list 1,50 --initial-birth-rate-range-list 4,10"
# NOTE use 100 sub procs for <1 milliong, otherwise 500 sub procs
# echo $bin $common --label new-constraints --version v2 --carry-cap-range-list 150,150 $pranges --n-replicates 1 --n-trials-list 5000:50000:500000:5000000 --simu-bundle-size 50 --epochs-list 100:250 --simu-extra-args=\"--n-max-procs 20\" --params-to-predict xscale:xshift:yscale
# echo $bin $common --label new-constraints --version v3 --carry-cap-range-list 150,150 $pranges --n-replicates 1 --n-trials-list 5000:50000:500000 --time-to-sampling-range-list 10,30 --simu-bundle-size 50 --epochs-list 50:250:1000 --prebundle-layer-cfg-list small:default:big --simu-extra-args=\"--n-max-procs 20\" --params-to-predict xscale:xshift:yscale
# pranges="--xscale-range-list 0.5,5 --xshift-range-list=-0.5,3 --yscale-range-list 1,50 --initial-birth-rate-range-list 4,10 --time-to-sampling-range-list 10,30 --carry-cap-range-list 100,300"
# also ran with epochs 250 and dl-bundle-size 1:10:50, and   
# echo $bin $common --label vary-all --version v0 $pranges --n-replicates 1 --n-trials-list 5000:50000:500000 --simu-bundle-size 50 --epochs-list 50:250:1000 --simu-extra-args=\"--n-max-procs 20\"
# pranges="--xscale-range-list 0.01,2 --xshift-range-list=-0.5,3 --yscale-range-list 1,50 --initial-birth-rate-range-list 4,10 --time-to-sampling-range-list 10,30 --carry-cap-range-list 100,300 --n-seqs-range-list 40,90"
# ppl="--params-to-predict-list xscale:xshift:yscale:xscale,xshift:xscale,yscale:xshift,yscale:xscale,xshift,yscale"
# echo $bin $common --label vary-all --version v1 $pranges --n-replicates 1 --n-trials-list 5000:50000:500000:670000 --simu-bundle-size-list 1:50:67 --dl-bundle-size-list 1:50:67 --zip-vars simu-bundle-size:dl-bundle-size --epochs-list 250 --simu-extra-args=\"--n-max-procs 20\" $ppl
# pranges="--xscale-range-list 0.01,2 --xshift-range-list=-0.5,3 --yscale-range-list 1,50 --time-to-sampling-range-list 17,25"
# echo $bin $common --label tree-infer-effect --version v2 --initial-birth-rate-range-list 0.5,1:1,5 --carry-cap-range-list 1000,1001 --init-population-list 1:100 --n-seqs-range-list 65,95 $pranges --n-replicates 1 --n-trials-list 75 --simu-extra-args=\"--n-max-procs 20 --init-population 100\"
# echo $bin $common --label retrain --version v0 --carry-cap-range-list 750,1500 $pranges --n-replicates 1 --n-trials-list 100:5000 --n-seqs-range-list 65,95 --simu-bundle-size-list 1:50:67 --dl-bundle-size-list 1:50:67 --zip-vars simu-bundle-size:dl-bundle-size --simu-extra-args=\"--n-max-procs 20 --init-population 100 --initial-birth-rate-range 0.1 0.5\"
# echo $bin $common --label const-leaves --version v1 --carry-cap-range-list 750,1500 $pranges --n-replicates 1 --n-trials-list 100:5000:50000 --simu-extra-args=\"--n-max-procs 20 --initial-birth-rate-range 0.1 0.5 --n-seqs-values 70 --init-population 128\"
# echo $bin $common --label const-leaves --version v2 --carry-cap-range-list 750,1500 $pranges --n-replicates 1 --n-trials-list 100:5000 --simu-bundle-size-list 1:10 --dl-bundle-size-list 1:10 --zip-vars simu-bundle-size:dl-bundle-size --simu-extra-args=\"--n-max-procs 20 --initial-birth-rate-range 0.1 0.5 --n-seqs-values 70 --init-population 128\"
# pranges="--xscale-range-list 0.01,2 --xshift-range-list=-0.5,3 --yscale-range-list 1,50 --time-to-sampling-range-list 17,25 --n-seqs-range-list 40,90"
# echo $bin $common --label retrain --version v1 --carry-cap-range-list 750,1500 $pranges --n-replicates 1 --n-trials-list 100:5000:50000 --simu-extra-args=\"--n-max-procs 20 --initial-birth-rate-range 0.1 0.5 --init-population 128\" --learning-rate-list 0.001:0.01 --epochs-list 500:5000 --batch-size-list 32:128
sxtra="--simu-extra-args=\"--n-max-procs 20 --initial-birth-rate-range 0.1 0.5 --init-population 128\""
dlargs="--learning-rate-list 0.001 --epochs-list 500 --batch-size-list 32"
pranges="--xscale-range-list 0.01,2 --xshift-range-list=-0.5,3 --yscale-range-list 0.5,50 --time-to-sampling-range-list 17,25 --n-seqs-range-list 40,90"
echo $bin $common --label for-data --version v0 --carry-cap-range-list 750,1500 $pranges --n-replicates 1 --n-trials-list 50000:100000:150000 $sxtra $dlargs --simu-bundle-size-list 1:2:3 --dl-bundle-size-list 1:2:3 --zip-vars simu-bundle-size:dl-bundle-size:n-trials --tree-inference-method iqtree
echo $bin $common --label data-mimic --version v2 --n-replicates 2 --n-trials-list 120 --simu-extra-args=\"--carry-cap-values 1500 --xscale-values 0.8 --xshift-values 1.6 --yscale-values 0.75 --initial-birth-rate-range 0.1 0.5 --n-seqs-range 60 95 --time-to-sampling-values 20 --init-population 128\" --n-sub-procs 60 --tree-inference-method iqtree
# echo $bin $common --label match-data --version v1 --n-replicates 2 --n-trials-list 120 --simu-extra-args=\"--carry-cap-values 1500 --xscale-values 0.8 --xshift-values 1.6 --yscale-values 0.75 --initial-birth-rate-range 0.4 0.5 --n-seqs-range 40 90 --time-to-sampling-values 20 --init-population 32\" --n-sub-procs 60 --tree-inference-method iqtree
exit 0

# dld=/fh/fast/matsen_e/dralph/partis/gcdyn/retrain/v1/n-trials-50000/epochs-500/batch-size-32/learning-rate-0.001/dl-infer  # old (but still fine) model
datadir=/fh/fast/matsen_e/data/taraki-gctree-2021-10/beast-processed-data/v4  # actual data (beast version)
# datadir=/fh/fast/matsen_e/data/taraki-gctree-2021-10/iqtree-processed-data/v2  # actual data (iqtree version)
# dlabel=data-iqtree-trained-v0 #-iqtree-data-v0
datadir=/fh/fast/matsen_e/data/taraki-gctree-2021-10/beast-processed-data/bst-simu-v1  # beast-processed simu
dlabel=bst-simu-v1
dsl=all-trees:d15-trees:d20-trees
dldstr=dl-infer/iqtree

# datadir=/fh/fast/matsen_e/dralph/partis/gcdyn/data-mimic/v1/seed-0
# dlabel=infer-on-data-mimic-v1
# dsl=simu:xxx
# dldstr=dl-infer #/iqtree

bargs="--base-outdir /fh/fast/matsen_e/dralph/partis/gcdyn --data-dir $datadir"
for bsize in 1 2 3; do
    let ntrial="50000 * $bsize"
    dld=/fh/fast/matsen_e/dralph/partis/gcdyn/for-data/v0/n-trials-$ntrial/simu-bundle-size-$bsize/dl-bundle-size-$bsize/$dldstr
    echo ./projects/cf-gcdyn.py --actions data --n-max-procs 5 --n-sub-procs 100 $bargs --label $dlabel --version bundle-size-$bsize --dry --dl-model-dir $dld --dl-bundle-size-list $bsize --data-samples-list $dsl
done
exit 0
