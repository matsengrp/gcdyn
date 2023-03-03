#!/usr/bin/env python2
import sys
import csv
csv.field_size_limit(sys.maxsize)  # make sure we can write very large csv fields
import os
import glob
import argparse
import colored_traceback.always

# if you move this script, you'll need to change this method of getting the imports
partis_dir = '%s/work/partis' % os.getenv('HOME') #os.path.dirname(os.path.realpath(__file__)).replace('/bin', '')
sys.path.insert(1, partis_dir + '/python')

import utils
import glutils
import plotting
import hutils

odir = '/fh/fast/matsen_e/dralph/gcdyn/tmp-out'
abdfn = '%s/abundances.csv' % odir

# ----------------------------------------------------------------------------------------
def parse_name(fn):  # convert .fa name to mouse #, etc
    fn = os.path.basename(fn)
    # fn = 'annotated-PR-1-10-11-LA-122-GC.fasta'
    nlist = fn.split('-')
    if len(nlist) != 8:
        raise Exception('fn should have 7 - chars, but got %d: %s' % (fn.count('-'), fn))
    assert nlist[:2] == ['annotated', 'PR']
    flowcell = '-'.join(nlist[2:4])
    mouse = int(nlist[4])
    ln_loc = nlist[5]
    ln_id = nlist[6]
    assert nlist[7] == 'GC.fasta'
    return mouse, flowcell, ln_loc, ln_id

# ----------------------------------------------------------------------------------------
def bstr(mouse, flowcell, ln_loc, ln_id):
    return '-'.join([str(mouse), flowcell, ln_loc, ln_id])

# ----------------------------------------------------------------------------------------
def calc_abdn():
    skipped_mice, kept_mice = set(), set()
    final_fnames = []
    for fasta_path in glob.glob('%s/*.fasta'%args.indir):
        mouse, flowcell, ln_loc, ln_id = parse_name(fasta_path)
        if args.mice is not None and mouse not in args.mice:
            skipped_mice.add(mouse)
            continue
        kept_mice.add(mouse)
        final_fnames.append(fasta_path)
    if len(skipped_mice) > 0:
        print '    skipped %d mice: %s' % (len(skipped_mice), ' '.join(str(s) for s in sorted(skipped_mice)))
    print '    kept %d samples from %d mice: %s' % (len(final_fnames), len(kept_mice), ' '.join(str(s) for s in sorted(kept_mice)))
    
    cmd = 'python scripts/abundance.py %s --outdir %s' % (' '.join(final_fnames), odir)
    utils.simplerun(cmd)

# ----------------------------------------------------------------------------------------
def plot():
    with open(abdfn) as afile:
        reader = csv.DictReader(afile)
        plotvals = {k : {} for k in reader.fieldnames if k!=''}
        for line in reader:
            abn = int(line[''])  # i can't figure out how to set this column label in the other script
            for bn in plotvals:
                plotvals[bn][abn] = int(line[bn])

    hists = []
    for bn in plotvals:
        htmp = hutils.make_hist_from_dict_of_counts(plotvals[bn], 'int', bn)
        hists.append(htmp)

    mhist = plotting.make_mean_hist(hists)
    ybounds = [0.9 * mhist.get_minimum(exclude_empty=True), 1.1 * mhist.get_maximum()]
    yticks, yticklabels = plotting.get_auto_y_ticks(ybounds[0], ybounds[1])
    ybounds = [yticks[0], yticks[-1]]
    mhist.fullplot(odir, 'abdn', pargs={'remove_empty_bins' : True}, fargs={'xbounds' : [0.5, mhist.xmax+0.5], 'ybounds' : ybounds, 'yticks' : yticks, 'yticklabels' : yticklabels,
                                                                            'xlabel' : 'abundance', 'ylabel' : 'N seqs\nmean+/-std, %d GCs'%len(hists), 'log' : 'y', 'xticks' : list(range(int(mhist.xmax)+1))})
    # plotting.draw_no_root(None, plotdir=odir, plotname='abdn', more_hists=hists, log='xy')

# ----------------------------------------------------------------------------------------
ustr = """
./projects/gcdyn/scripts/plot-abdn.py --indir /fh/fast/matsen_e/dralph/gcdyn/gcreplay-observed
"""
parser = argparse.ArgumentParser(usage=ustr)
parser.add_argument('--indir')
parser.add_argument('--outdir')
parser.add_argument('--mice', default=[1, 2, 3, 4, 5, 6], help='restrict to these mouse numbers')
args = parser.parse_args()

# calc_abdn()
plot()
