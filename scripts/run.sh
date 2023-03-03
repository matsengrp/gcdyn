#!/bin/bash

vsn=test-v0
ddir=/fh/fast/matsen_e/dralph/gcdyn/gcreplay-observed
odir=/fh/fast/matsen_e/dralph/gcdyn/$vsn

echo ./scripts/plot-abdn.py --data-dir $ddir --simu-dir $odir/simu --outdir $odir/plots
# echo python scripts/multi-simulation.py --outdir $odir/simu --n-seqs 70 --n-trials 1

# echo ./bin/compare-plotdirs.py --file-glob-str='abdn.csv' --outdir $odir/comparisons --plotdirs $odir/data-plots:$odir/simu-plots --names data:simu --log y

