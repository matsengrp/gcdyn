#!/usr/bin/env python
# fmt: off
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
def get_affinity(nuc_seq, name='', debug=False):
    # ----------------------------------------------------------------------------------------
    def get_mutations(naive_aa, aa, pos_map, chain_annotation):
        assert len(naive_aa) == len(aa)
        return [f"{aa1}{pos_map[pos]}{chain_annotation}{aa2}"
                for pos, (aa1, aa2) in enumerate(zip(naive_aa, aa))
                if aa1 != aa2]
    # ----------------------------------------------------------------------------------------
    def ltranslate(tseq):
        if 'Bio.Seq' not in sys.modules:  # import is frequently slow af
            from Bio.Seq import Seq
        bseq = sys.modules['Bio.Seq']
        return str(bseq.Seq(tseq).translate())
    # ----------------------------------------------------------------------------------------
    aa_seqs = {'h' : ltranslate(nuc_seq[:336]),
               'l' : ltranslate(nuc_seq[336:])}
    muts = {c : get_mutations(naive_seqs[c], aa_seqs[c], pos_maps[c], "(%s)"%c.upper()) for c in 'hl'}
    has_stops = [any("*" in x for x in muts[c]) for c in 'hl']
    if any(has_stops):  # should really be "affinity" or "delta bind" rather than kd_val
        kd_val = np.nan
        kdstr = utils.color('red', ' stop')
    else:
        kd_val = dms_df.delta_bind_CGG[muts['h'] + muts['l']].sum()
        kdstr = '%5.2f'%kd_val
    if debug:
        print('    %15s %s  %3d%3d  %3d' % (name, kdstr, len(muts['h']), len(muts['l']), len(muts['h']+muts['l'])), end='')
        if args.check_gct_kd and node.taxon.label in gct_kd_vals:
            gctval = gct_kd_vals[node.taxon.label]
            gctstr = utils.color('blue', '-', width=5) if gctval is None else '%5.2f'%gctval
            print('    %s %s' % (gctstr, 'ok' if abs(kd_val-gctval) / gctval < 0.001 else utils.color('red', 'nope')), end='')
        print()
    return kd_val, len(muts['h'] + muts['l'])

# ----------------------------------------------------------------------------------------
def get_etree(dtree, debug=False):
    etree, enodes, lmetafos = None, {}, []
    for node in dtree.preorder_node_iter():
        nlab = node.taxon.label
        assert nlab not in enodes
        enodes[nlab] = bdms.TreeNode(t=node.distance_from_root(), dist=node.edge.length, name=nlab)  # note that the dtree distances are times, so all leaves are the same distance to root (this sets root node distance to None)
        if node is dtree.seed_node:
            assert etree is None
            etree = enodes[nlab]
        enodes[nlab].x, n_muts = get_affinity(node.nuc_seq, name=nlab, debug=args.debug)
        lmetafos.append({'name' : nlab, 'affinity' : enodes[nlab].x, 'n_muts' : n_muts})

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
    return etree, lmetafos

# ----------------------------------------------------------------------------------------
def read_gct_kd():
    print('  reading kd values from %s' % args.gct_kd_fname)
    kd_vals = {}
    with open(args.gct_kd_fname) as afile:  # note that I also use these kd vals in partis/bin/read-gctree-output.py, but there I'm using gctree output so it's a bit different (and is processed through datascripts/taraki-gctree-2021-10)
        reader = csv.DictReader(afile)
        for line in reader:
            kdv = line['delta_bind_CGG_FVS_additive']
            kd_vals[line['ID_HK']] = float(kdv) if kdv != '' else None
    return kd_vals

# ----------------------------------------------------------------------------------------
def get_mut_info():
    print('    reading variant scores from %s' % args.variant_score_fname)
    dms_df = pd.read_csv(args.variant_score_fname, index_col="mutation", dtype=dict(position_IMGT=pd.Int16Dtype())) #"https://media.githubusercontent.com/media/jbloomlab/Ab-CGGnaive_DMS/main/results/final_variant_scores/final_variant_scores.csv"
    dms_df = dms_df[dms_df.chain != "link"]  # remove linker sites
    dms_df["WT"] = dms_df.wildtype == dms_df.mutant  # add indicator for wildtype data
    assert dms_df.position_IMGT.max() < 1000
    dms_df["site"] = [f"{chain}-{str(pos).zfill(3)}" for chain, pos in zip(dms_df.chain, dms_df.position_IMGT)]

    print('    reading cgg naive position info from %s' % args.cgg_naive_sites_fname)
    pos_df = pd.read_csv(args.cgg_naive_sites_fname, dtype=dict(site=pd.Int16Dtype()), index_col="site_scFv")  # "https://raw.githubusercontent.com/jbloomlab/Ab-CGGnaive_DMS/main/data/CGGnaive_sites.csv"
    naive_seqs = {c : "".join(pos_df.query("chain == '%s'"%c.upper()).amino_acid) for c in 'hl'}
    naive_seqs['l'] = naive_seqs['l'][:-1]  # trim the last character, for some reason
    pos_maps = {c : pos_df.loc[pos_df.chain == c.upper(), "site"].reset_index(drop=True) for c in 'hl'}

    return dms_df, naive_seqs, pos_maps

# ----------------------------------------------------------------------------------------
example_results_dir = '/fh/fast/matsen_e/shared/replay-related/jareds-replay-fork/gcreplay/nextflow/results'
example_beastdir = '%s/2023-05-18-beast/beast/btt-PR-1-1-1-LB-20-GC' % example_results_dir
helpstr = """
Read Beast results from xml and history.trees files, add phenotype (affinity/kd) info from additive DMS-based
affinity model, then write sequences to fasta, tree to newick, and affinity to yaml.
Example usage:
    parse-beast.py --test --outdir <outdir>
    parse-beast.py --outdir <outdir> --xml-fname <beastdir>/beastgen.xml --trees-fname <beastdir>/*.history.trees --debug --check
"""
class MultiplyInheritedFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass
formatter_class = MultiplyInheritedFormatter
parser = argparse.ArgumentParser(formatter_class=MultiplyInheritedFormatter, description=helpstr)
parser.add_argument("--xml-fname", help='beast xml file')
parser.add_argument("--trees-fname", help='beast .history.trees file')
parser.add_argument("--outdir", required=True)
parser.add_argument("--max-leaf-count", type=int, default=100)
parser.add_argument("--igk-idx", type=int, default=336, help='zero-based index of first igk position in smooshed-together igh+igk sequence')
parser.add_argument("--debug", action='store_true')
parser.add_argument("--check-gct-kd", action='store_true')
parser.add_argument("--gct-kd-fname", default='%s/latest/merged-results/observed-seqs.csv'%example_results_dir, help='old additive kd numbers from a gctree run')
parser.add_argument("--test", action='store_true', help='if set, run on example beast files (see other args)')
parser.add_argument("--variant-score-fname", default='projects/gcdyn/experiments/final_variant_scores.csv')
parser.add_argument("--cgg-naive-sites-fname", default='projects/gcdyn/experiments/CGGnaive_sites.csv')
args = parser.parse_args()

if args.test:
    print('  --test: running with results from example dir %s' % example_results_dir)
    args.xml_fname = '%s/beastgen.xml' % example_beastdir
    args.trees_fname = '%s/beastannotated-PR-1-1-1-LB-20-GC_with_time.history.trees' % example_beastdir
    print('    --xml-fname: %s' % args.xml_fname)
    print('    --trees-fname: %s' % args.trees_fname)
else:
    if args.xml_fname is None or args.trees_fname is None:
        raise Exception('both --xml-fname and --trees-fname must be set unless running with --test')

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

dms_df, naive_seqs, pos_maps = get_mut_info()

single_treefname = '%s/single-tree.history.trees' % args.outdir
print('  writing first tree from beast history file to new file %s' % single_treefname)
subprocess.check_call('grep -v \'^tree STATE_[1-9][^ ]\' %s >%s' % (args.trees_fname, single_treefname), shell=True)

print('  loading trees from %s' % single_treefname)
sys.stdout.flush()
res_gen, rmd_sites = beast_loader.load_beast_trees(args.xml_fname, single_treefname)
print('    finished loading trees')
sys.stdout.flush()

if args.check_gct_kd:
    gct_kd_vals = read_gct_kd()

sstats, encoded_trees, lmetafos = [], [], []
if args.debug:
    print('                               N muts')
    print('              name    kd     h  l   h+l')
for itr, dtree in enumerate(res_gen):
    with open('%s/seqs.fa' % args.outdir, 'w') as sfile:
        inodelabel = 0
        for node in dtree.preorder_node_iter():
            if node.taxon is None:
                node.taxon = dendropy.Taxon('node-%d' % inodelabel)
                inodelabel += 1
            node.taxon.label = node.taxon.label.replace('@20', '')  # not sure what this is for
            node.nuc_seq = node.observed_cg.to_sequence() if node.is_leaf() else node.cg.to_sequence()
            sfile.write('>%s\n%s\n' % (node.taxon.label, node.nuc_seq))
    with open('%s/tree.nwk' % args.outdir, 'w') as tfile:
        tfile.write('%s\n' % dtree.as_string(schema='newick').strip())

    etree, tr_lmfos = get_etree(dtree, debug=args.debug)
    brlen, sctree = encode.scale_tree(etree)
    enc_tree = encode.encode_tree(etree, max_leaf_count=args.max_leaf_count, dont_scale=True)

    sstats.append({'tree' : itr, 'mean_branch_length' : brlen, 'total_branch_length' : sum(n.dist for n in etree.iter_descendants())})
    encoded_trees.append(enc_tree)
    lmetafos += tr_lmfos

print('  writing %d trees to %s' % (len(encoded_trees), args.outdir))
encode.write_trees(encode.output_fn(args.outdir, 'encoded-trees', None), encoded_trees)
encode.write_sstats(encode.output_fn(args.outdir, 'summary-stats', None), sstats)
encode.write_leaf_meta(encode.output_fn(args.outdir, 'leaf-meta', None), lmetafos)
