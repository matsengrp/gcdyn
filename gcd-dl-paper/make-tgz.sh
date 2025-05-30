#!/bin/bash

datadir=/fh/fast/matsen_e/data/taraki-gctree-2021-10/iqtree-processed-data/v4
fsd=/fh/fast/matsen_e/dralph/partis/gcdyn
outdir=$fsd/zenodo

# indirs="test/v0"
# indirs="$datadir/combo-trees $fsd/for-data/v6 $fsd/data-iqtree-trained-iqtree-data/v8"
indirs=$fsd/for-data/v6
for indir in $indirs; do
    ssh quokka "cd $indir && tar czf $outdir/`echo $indir | sed s@$fsd/@@ | sed s@$datadir/@@ | sed 's@/@-@g'`.tgz ." &
done
