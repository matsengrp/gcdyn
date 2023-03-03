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
def filter_mice(in_fns):
    skipped_mice, kept_mice = set(), set()
    fn_fns = []
    for fasta_path in in_fns:
        mouse, flowcell, ln_loc, ln_id = parse_name(fasta_path)
        if args.mice is not None and mouse not in args.mice:
            skipped_mice.add(mouse)
            continue
        kept_mice.add(mouse)
        fn_fns.append(fasta_path)
    if len(skipped_mice) > 0:
        print '    skipped %d mice: %s' % (len(skipped_mice), ' '.join(str(s) for s in sorted(skipped_mice)))
    print '    kept %d samples from %d mice: %s' % (len(fn_fns), len(kept_mice), ' '.join(str(s) for s in sorted(kept_mice)))
    return fn_fns

# ----------------------------------------------------------------------------------------
def calc_abdn(indir, label, is_simu):
    fn_fns = glob.glob('%s/*.fasta'%indir)
    if not is_simu:
        fn_fns = filter_mice(fn_fns)
    cmd = 'python scripts/abundance.py %s --min-seqs 70 --max-seqs 70 --outdir %s/%s' % (' '.join(fn_fns), args.outdir, label)
    utils.simplerun(cmd)

# ----------------------------------------------------------------------------------------
def hargs(htmp):
    xbounds = [0.5, htmp.xmax+0.5]
    xticks = list(range(1, int(htmp.xmax)+1, 2))
    ybounds = [0.9 * htmp.get_minimum(exclude_empty=True), 1.1 * htmp.get_maximum()]
    yticks, yticklabels = plotting.get_auto_y_ticks(ybounds[0], ybounds[1])
    ybounds = [yticks[0], yticks[-1]]
    return xbounds, ybounds, xticks, yticks, yticklabels

# ----------------------------------------------------------------------------------------
def plot(base_outdir, label):
    with open('%s/%s/abundances.csv'%(base_outdir, label)) as afile:
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
    mhist.title = '%s (%d GCs)' % (label, len(hists))
    mhist.xtitle = 'abundance'
    mhist.ytitle = 'N seqs\nmean+/-std, %d GCs' % len(hists)
    xbounds, ybounds, xticks, yticks, yticklabels = hargs(mhist)
    fn = mhist.fullplot(base_outdir+'/'+label, 'abdn', pargs={'remove_empty_bins' : True}, fargs={'xbounds' : xbounds, 'ybounds' : ybounds, 'yticks' : yticks, 'yticklabels' : yticklabels,
                                                                                                  'xlabel' : mhist.xtitle, 'ylabel' : mhist.ytitle, 'log' : 'y', 'xticks' : xticks})
    fnames[0].append(fn)
    return mhist

# ----------------------------------------------------------------------------------------
def compare_plots(base_outdir, hists, labels):
    utils.mkdir(base_outdir+'/comparisons')
    xbounds, ybounds, xticks, yticks, yticklabels = hargs(hists[0])
    ytitle = '%s\nmean +/- std' % hists[0].ytitle.split('\n')[0]
    fn = plotting.draw_no_root(None, plotdir=base_outdir+'/comparisons', plotname='abdn', more_hists=hists, log='y', xtitle=hists[0].xtitle, ytitle=ytitle,
                               bounds=xbounds, ybounds=ybounds, xticks=xticks, yticks=yticks, yticklabels=yticklabels, errors=True, linewidths=[4, 3], plottitle='')
    fnames[0].append(fn)

# ----------------------------------------------------------------------------------------
ustr = """
./scripts/plot-abdn.py --data-dir <path with fastas> --simu-dir <path with fastas>
"""
parser = argparse.ArgumentParser(usage=ustr)
parser.add_argument('--data-dir')
parser.add_argument('--simu-dir')
parser.add_argument('--outdir')
parser.add_argument('--mice', default=[1, 2, 3, 4, 5, 6], help='restrict to these mouse numbers')
parser.add_argument('--is-simu', action='store_true')
args = parser.parse_args()

dlabels = []
if args.data_dir is not None:
    dlabels.append([args.data_dir, 'data'])
if args.simu_dir is not None:
    dlabels.append([args.simu_dir, 'simu'])

hists, fnames = [], [[]]
for idir, tlab in dlabels:
    calc_abdn(idir, tlab, is_simu=tlab=='simu')
    hists.append(plot(args.outdir, tlab))

compare_plots(args.outdir, hists, [l for _, l in dlabels])
plotting.make_html(args.outdir, fnames=fnames)
