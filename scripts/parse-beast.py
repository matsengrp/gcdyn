#!/usr/bin/env python
import numpy as np
import argparse
import os
import sys
from sklearn import preprocessing
import subprocess
import dendropy
import colored_traceback.always  # noqa: F401
import time
import pickle
import pandas as pd
import random
import csv
import copy
import collections

from gcdyn import bdms, utils, encode
from historydag import beast_loader

# ----------------------------------------------------------------------------------------
def get_etree(dtree, debug=False):
    etree, enodes = None, {}
    for node in dtree.preorder_node_iter():
        assert node.taxon.label not in enodes
        enodes[node.taxon.label] = bdms.TreeNode(t=node.distance_from_root(), dist=0, name=node.taxon.label)  # note that the dtree distances are times, so all leaves are the same distance to root
        if node is dtree.seed_node:
            assert etree is None
            etree = enodes[node.taxon.label]
    for node in dtree.preorder_node_iter():
        for cnode in node.child_node_iter():
            enodes[node.taxon.label].add_child(enodes[cnode.taxon.label])
    if etree is None:
        raise Exception('didn\'t find root node')
    if debug:
        print('dendro')
        print(dtree.as_ascii_plot(width=150, plot_metric='depth', show_internal_node_labels=True))
        print('ete3')
        print(etree.get_ascii(show_internal=True))
    return etree

# ----------------------------------------------------------------------------------------
bd = '/fh/fast/matsen_e/shared/replay-related/jareds-replay-fork/gcreplay/nextflow/results/2023-05-18-beast/beast/btt-PR-1-1-1-LB-20-GC'
parser = argparse.ArgumentParser()
parser.add_argument("xml_fname", default='%s/beastgen.xml'%bd)
parser.add_argument("trees_fname", default='%s/beastannotated-PR-1-1-1-LB-20-GC_with_time.history.trees'%bd)
parser.add_argument("outdir")
parser.add_argument("--max-leaf-count", type=int, default=100)
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

single_treefname = '%s/single-tree.history.trees' % args.outdir
subprocess.check_call('grep -v \'^tree STATE_[1-9][^ ]\' %s >%s' % (args.trees_fname, single_treefname), shell=True)

print('  loading trees from %s' % single_treefname)
sys.stdout.flush()
res_gen, rmd_sites = beast_loader.load_beast_trees(args.xml_fname, single_treefname)
print('    finished loading trees')
sys.stdout.flush()

for dtree in res_gen:
    with open('%s/seqs.fa' % args.outdir, 'w') as sfile:
        inodelabel = 0
        for node in dtree.preorder_node_iter():
            if node.taxon is None:
                node.taxon = dendropy.Taxon('node-%d' % inodelabel)
                inodelabel += 1
            nseq = node.observed_cg.to_sequence() if node.is_leaf() else node.cg.to_sequence()
            sfile.write('>%s\n%s\n' % (node.taxon.label, nseq))
    with open('%s/tree.nwk' % args.outdir, 'w') as tfile:
        tfile.write('%s\n' % dtree.as_string(schema='newick').strip())

    etree = get_etree(dtree)
    brlen, sctree = encode.scale_tree(etree)
    enc_tree = encode.encode_tree(etree, max_leaf_count=args.max_leaf_count, dont_scale=True)
