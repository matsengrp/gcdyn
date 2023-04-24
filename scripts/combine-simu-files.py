import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from Bio import SeqIO
# import colored_traceback.always  # need to add this to installation stuff, i'm not sure how to do it atm
import dill

from gcdyn import bdms, gpmap, mutators, poisson, utils
from experiments import replay
from colors import color

# ----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('infiles', nargs='+', help='list of files with dill pickled gcdyn simulation')
parser.add_argument('--outfile', required=True, help='output file name for concatenated simulation info (response functions and trees)')
args = parser.parse_args()

simfo = {'trees' : [], 'responses' : []}  # 'responses': pair of [birth, death] responses
for ifn in args.infiles:
    with open(ifn, 'rb') as dfile:
        dfo = dill.load(dfile)
        for etree in dfo['trees']:
            simfo['trees'].append(etree)
            simfo['responses'].append([dfo[r+'-response'] for r in ['birth', 'death']])

print('  writing %d trees from %d files to %s' % (len(simfo['trees']), len(args.infiles), args.outfile))
if not os.path.exists(os.path.dirname(args.outfile)):
    os.makedirs(os.path.dirname(args.outfile))
with open(args.outfile, 'wb') as pfile:
    dill.dump(simfo, pfile)

with open(args.outfile, 'rb') as pfile:
    dfo = dill.load(pfile)
    treestrs = [str(len(list(t.iter_leaves()))) for t in dfo['trees']]
    birthstrs, deathstrs = zip(*[rp for rp in dfo['responses']])
    print('    checking info in outfile: %d trees with leaf counts: %s' % (len(dfo['trees']), ' '.join(treestrs)))
    print('        distinct response fcns:  birth %d  death %d' % (len(set(birthstrs)), len(set(deathstrs))))
