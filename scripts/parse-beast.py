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
# dfs = (pd.read_csv("projects/gcreplay/nextflow/data/10x/Timecourse_Novaseqvdj/Data/AV1_VDJ_res/filtered_contig_annotations.csv"),
#        pd.read_csv("projects/gcreplay/nextflow/data/10x/Timecourse_Novaseqvdj/Data/AV2_VDJ_res/filtered_contig_annotations.csv"),
#        pd.read_csv("projects/gcreplay/nextflow/data/10x/Timecourse_Novaseqvdj/Data/AV3_VDJ_res/filtered_contig_annotations.csv")
#        )
dfs = (pd.read_csv("projects/gcreplay/nextflow/data/10x/tmp/filtered_contig_annotations.csv"),
       pd.read_csv("projects/gcreplay/nextflow/data/10x/tmp/filtered_contig_annotations-1.csv"),
       pd.read_csv("projects/gcreplay/nextflow/data/10x/tmp/filtered_contig_annotations-2.csv"),
       )
for i in range(3):
    dfs[i]['library'] = i + 1
df = pd.concat(dfs).reset_index(drop=True)
df = df.groupby("barcode").filter(lambda x: len(x.index) == 2)

# read dms info
dms_df = pd.read_csv('projects/gcdyn/experiments/final_variant_scores.csv', index_col="mutation", dtype=dict(position_IMGT=pd.Int16Dtype())) #"https://media.githubusercontent.com/media/jbloomlab/Ab-CGGnaive_DMS/main/results/final_variant_scores/final_variant_scores.csv"
dms_df = dms_df[dms_df.chain != "link"]  # remove linker sites
dms_df["WT"] = dms_df.wildtype == dms_df.mutant  # add indicator for wildtype data
assert dms_df.position_IMGT.max() < 1000
dms_df["site"] = [f"{chain}-{str(pos).zfill(3)}" for chain, pos in zip(dms_df.chain, dms_df.position_IMGT)]

# read position info
pos_df = pd.read_csv('projects/gcdyn/experiments/CGGnaive_sites.csv', dtype=dict(site=pd.Int16Dtype()), index_col="site_scFv")  # "https://raw.githubusercontent.com/jbloomlab/Ab-CGGnaive_DMS/main/data/CGGnaive_sites.csv"

naive_H = "".join(pos_df.query("chain == 'H'").amino_acid)
naive_L = "".join(pos_df.query("chain == 'L'").amino_acid)[:-1]

# add full-length aa/nt seqs to 10x df
df["aa_seq"] = df.fwr1 + df.cdr1 + df.fwr2 + df.cdr2 + df.fwr3 + df.cdr3 + df.fwr4
df["nt_seq"] = df.fwr1_nt + df.cdr1_nt + df.fwr2_nt + df.cdr2_nt + df.fwr3_nt + df.cdr3_nt + df.fwr4_nt

# Filter to only sequences with the expected length of each chain
df = df.loc[((df.chain == 'IGH') & (df.aa_seq.str.len() == len(naive_H))) | ((df.chain == 'IGK') & (df.aa_seq.str.len() == len(naive_L)))].reset_index(drop=True)

def get_mutations(naive_aa, aa, pos_map, chain_annotation):
    assert len(naive_aa) == len(aa)
    return [f"{aa1}{pos_map[pos]}{chain_annotation}{aa2}"
            for pos, (aa1, aa2) in enumerate(zip(naive_aa, aa))
            if aa1 != aa2]

pos_map_H = pos_df.loc[pos_df.chain == "H", "site"].reset_index(drop=True)
pos_map_L = pos_df.loc[pos_df.chain == "L", "site"].reset_index(drop=True)

for idx in df.index:
    print(idx, df.chain[idx])
    if df.chain[idx] == "IGH":
         mutations = get_mutations(naive_H, df.aa_seq[idx], pos_map_H, "(H)")
    elif df.chain[idx] == "IGK":
         mutations = get_mutations(naive_L, df.aa_seq[idx], pos_map_L, "(L)")
    else:
         print(f"skipping unexpected chain length {df.chain[idx]} {len(df.aa_seq[idx])}")
         continue

    df.loc[idx, "mutations"] = ",".join(mutations)
    df.loc[idx, "n_mutations"] = len(mutations)
    df.loc[idx, "delta_bind_CGG"] = dms_df.delta_bind_CGG[mutations].sum()
    df.loc[idx, "delta_expr"] = dms_df.delta_expr[mutations].sum()
    df.loc[idx, "delta_psr"] = dms_df.delta_psr[mutations].sum()

df.n_mutations = df.n_mutations.astype("Int64")

sys.exit()


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
