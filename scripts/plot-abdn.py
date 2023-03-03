#!/usr/bin/env python
import sys
import csv
csv.field_size_limit(sys.maxsize)  # make sure we can write very large csv fields
import os
import argparse
import colored_traceback.always

# if you move this script, you'll need to change this method of getting the imports
partis_dir = os.getcwd() #os.path.dirname(os.path.realpath(__file__)).replace('/bin', '')
sys.path.insert(1, partis_dir + '/python')

import utils
import glutils
import plotting
import hutils

odir = '/fh/fast/matsen_e/dralph/gcdyn/tmp-out'
abdfn = '%s/abundances.csv' % odir

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
mhist.fullplot(odir, 'abdn', pargs={'remove_empty_bins' : True}, fargs={'xbounds' : [0.5, mhist.xmax+0.5], 'xlabel' : 'abundance', 'ylabel' : 'N seqs\nmean+/-std, %d GCs'%len(hists), 'log' : 'y', 'xticks' : list(range(int(mhist.xmax)+1))})
# plotting.draw_no_root(None, plotdir=odir, plotname='abdn', more_hists=hists, log='xy')
