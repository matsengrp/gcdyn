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
import glob

from gcdyn import bdms, utils, encode
from historydag import beast_loader
partis_dir = os.path.dirname(os.path.realpath(__file__)).replace('/projects/gcdyn/scripts', '')
sys.path.insert(1, partis_dir) # + '/python')
import python.treeutils as treeutils
import python.utils as utils
import python.datautils as datautils

# ----------------------------------------------------------------------------------------
def get_affinity(nuc_seq, name, kd_checks, debug=False):
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
        kd_val = dms_df.delta_bind_CGG.min()
        kdstr = utils.color('red', ' stop')
    else:
        kd_val = dms_df.delta_bind_CGG[muts['h'] + muts['l']].sum()
        kdstr = '%5.2f' % kd_val
    if debug:
        print('    %18s %s  %3d%3d  %3d' % (name, kdstr, len(muts['h']), len(muts['l']), len(muts['h']+muts['l'])), end='')
        if args.check_gct_kd and name in gct_kd_vals:
            gctval = gct_kd_vals[name]
            gctstr = utils.color('blue', '-', width=5) if gctval is None else '%5.2f'%gctval
            fdiff = 0 if gctval==0 and kd_val==0 else abs((kd_val-gctval)) / max(abs(gctval), abs(kd_val))
            chkstr = 'ok' if fdiff < 0.001 else 'bad'
            kd_checks[chkstr] += 1
            print('    %s %s' % (gctstr, utils.color('red' if chkstr=='bad' else None, chkstr)), end='')
        print()
    return kd_val, len(muts['h'] + muts['l'])

# ----------------------------------------------------------------------------------------
def get_etree(dtree, idir, itr, kd_checks, debug=False):
    etree, enodes, lmetafos = None, {}, []
    for node in dtree.preorder_node_iter():
        nlab = node.taxon.label
        assert nlab not in enodes
        enodes[nlab] = bdms.TreeNode(t=node.distance_from_root(), dist=node.edge.length, name=nlab)  # note that the dtree distances are times, so all leaves are the same distance to root (this sets root node distance to None)
        if node is dtree.seed_node:
            assert etree is None
            etree = enodes[nlab]
        enodes[nlab].x, n_muts = get_affinity(node.nuc_seq, nlab, kd_checks, debug=debug)
        assert itr == 0  # would want to update tree-index
        lmetafos.append({'tree-index' : idir, 'name' : nlab, 'affinity' : enodes[nlab].x, 'n_muts' : n_muts})

    for node in dtree.preorder_node_iter():
        for cnode in node.child_node_iter():
            enodes[node.taxon.label].add_child(enodes[cnode.taxon.label])
    if etree is None:
        raise Exception('didn\'t find root node')
    if debug > 1:
        print('dendro')
        print(dtree.as_ascii_plot(width=150, plot_metric='depth', show_internal_node_labels=True))
        print('ete3')
        print(etree.get_ascii(show_internal=True))
    return etree, lmetafos

# ----------------------------------------------------------------------------------------
def read_gct_kd():
    gct_kd_fname = '%s/latest/merged-results/observed-seqs.csv' % args.input_dir
    print('  reading kd values from %s' % gct_kd_fname)
    kd_vals = {}
    with open(gct_kd_fname) as afile:  # note that I also use these kd vals in partis/bin/read-gctree-output.py, but there I'm using gctree output so it's a bit different (and is processed through datascripts/taraki-gctree-2021-10)
        reader = csv.DictReader(afile)
        for line in reader:
            kdv = line['delta_bind_CGG_FVS_additive']
            kd_vals[line['ID_HK']] = float(kdv) if kdv != '' else dms_df.delta_bind_CGG.min()
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
def read_single_dir(bstdir, idir):
    gclabel = os.path.basename(bstdir)
    gcodir = '%s/%s' % (baseoutdir, gclabel)
    if not os.path.exists(gcodir):
        os.makedirs(gcodir)

    single_treefname = '%s/single-tree.history.trees' % gcodir
    tfns = glob.glob('%s/beastannotated-*.history.trees' % bstdir)
    if len(tfns) != 1:
        raise Exception('expected one tree history file *.history.trees but got %d in %s' % (len(tfns), bstdir))
    if args.debug:
        print('    %s' % gclabel)
        print('        reading beast info from %s' % bstdir)
        print('             writing first tree from beast history file to new file %s' % single_treefname)
    # NOTE if you start reading more than one tree, it'll no longer really make sense to concat all the trees from all gc dirs together at the end
    subprocess.check_call('grep -v \'^tree STATE_[1-9][^ ]\' %s >%s' % (tfns[0], single_treefname), shell=True)

    if args.debug:
        print('          loading trees from %s' % single_treefname)
    res_gen, rmd_sites = beast_loader.load_beast_trees('%s/beastgen.xml'%bstdir, single_treefname)

    if args.debug:
        print('                               N muts')
        print('              name       kd     h  l   h+l')
    kd_checks = {'ok' : 0, 'bad' : 0}
    for itr, dtree in enumerate(res_gen):
        sofn = encode.output_fn(gcodir, 'seqs', None)
        if not os.path.exists(sofn):  # UGH (used to use the wrong file name)
            sofn = sofn.replace('fasta', 'fa')
        trsfos = []
        with open(sofn, 'w') as sfile:
            inodelabel = 0
            for node in dtree.preorder_node_iter():
                if node.taxon is None:
                    assert itr == 0  # if you start reading more than one tree, you [probably] need to change the node names so they tell you which tree they're from
                    node.taxon = dendropy.Taxon('%d-node-%d' % (idir, inodelabel))
                    inodelabel += 1
                else:
                    node.taxon.label = '%d-%s' % (idir, node.taxon.label)
                node.taxon.label = node.taxon.label.replace('@20', '').replace('@0', '')  # not sure what this is for
                node.nuc_seq = node.observed_cg.to_sequence() if node.is_leaf() else node.cg.to_sequence()
                sfile.write('>%s\n%s\n' % (node.taxon.label, node.nuc_seq))
                trsfos.append({'name' : node.taxon.label, 'seq' : node.nuc_seq})
        with open('%s/trees.nwk' % gcodir, 'w') as tfile:
            tfile.write('%s\n' % dtree.as_string(schema='newick').strip())

        etree, tr_lmfos = get_etree(dtree, idir, itr, kd_checks, debug=args.debug)
        brlen, sctree = encode.scale_tree(etree)
        enc_tree = encode.encode_tree(etree, max_leaf_count=args.max_leaf_count, dont_scale=True)

        assert itr == 0  # would want to change tree index ('tree' key) below
        sstats = {'tree' : idir, 'mean_branch_length' : brlen, 'total_branch_length' : sum(n.dist for n in etree.iter_descendants())}
        break  # seems to only ever be one tree in res_gen atm anyway

    print('    tree with %d leaf nodes' % len(list(dtree.leaf_node_iter())))

    if kd_checks['bad'] > 0:
        print('  %s %d / %d affinities don\'t match old values from gctree results' % (utils.color('yellow', 'warning'), kd_checks['bad'], sum(kd_checks.values())))

    if args.debug:
        print('      writing tree to %s' % gcodir)
    encode.write_trees(encode.output_fn(gcodir, 'encoded-trees', None), [enc_tree])
    encode.write_sstats(encode.output_fn(gcodir, 'summary-stats', None), [sstats])
    encode.write_leaf_meta(encode.output_fn(gcodir, 'meta', None), tr_lmfos)

    all_info['encoded_trees'].append(enc_tree)
    all_info['sstats'].append(sstats)
    all_info['lmetafos'].append(tr_lmfos)
    all_info['seqfos'].append(trsfos)
    all_info['dtrees'].append(dtree)
    all_info['metafos'].append(rpmeta[datautils.fix_btt_id(gclabel)])

# ----------------------------------------------------------------------------------------
def subset_info(timepoint):
    n_kept, n_tot = 0, 0
    returnfo = {k : [] for k in infokeys}
    if len(set(len(vlist) for vlist in all_info.values())) > 1:
        raise Exception('different length vlists: %s' % set(len(vlist) for vlist in all_info.values()))
    for itree in range(len(all_info['encoded_trees'])):
        n_tot += 1
        mfo = all_info['metafos'][itree]
        tps = mfo['time']
        if tps != timepoint:
            continue
        n_kept += 1
        for tk in infokeys:
            returnfo[tk].append(all_info[tk][itree])
    print('  %s: kept %d / %d trees' % (timepoint, n_kept, n_tot))
    return returnfo

# ----------------------------------------------------------------------------------------
def write_output(outdir, infolists):
    # concatenate all the trees read from each dir NOTE this only really makes sense if you only read *one* from each dir (each dir corresponds to one gc, but has lots of sampled beast trees for that gc)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print('  writing %d trees to %s' % (len(infolists['encoded_trees']), outdir))
    encode.write_trees(encode.output_fn(outdir, 'encoded-trees', None), infolists['encoded_trees'])
    encode.write_sstats(encode.output_fn(outdir, 'summary-stats', None), infolists['sstats'])
    encode.write_leaf_meta(encode.output_fn(outdir, 'meta', None), [lfo for lflist in infolists['lmetafos'] for lfo in lflist])
    utils.write_fasta(encode.output_fn(outdir, 'seqs', None), [s for slist in infolists['seqfos'] for s in slist])
    with open('%s/trees.nwk' % outdir, 'w') as tfile:
        for dtree in infolists['dtrees']:
            tfile.write('%s\n' % dtree.as_string(schema='newick').strip())

# ----------------------------------------------------------------------------------------
helpstr = """
Read Beast results from xml and history.trees files, add phenotype (affinity/kd) info from additive DMS-based
affinity model, then write sequences to fasta, tree to newick, and affinity to yaml.
Example usage:
    beast-parse.py --input-dir <input-dir> --beast-version <bvsn> --output-version <ovsn> --debug 1 --check
(see also datascripts/meta/taraki-gctree-2021-10/run.sh)
"""
class MultiplyInheritedFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass
formatter_class = MultiplyInheritedFormatter
parser = argparse.ArgumentParser(formatter_class=MultiplyInheritedFormatter, description=helpstr)
parser.add_argument("--input-dir", default='/fh/fast/matsen_e/shared/replay-related/jareds-replay-fork/gcreplay/nextflow/results', help='base input dir with beast results and dms affinity info')
parser.add_argument('--gcreplay-dir', default='/fh/fast/matsen_e/data/taraki-gctree-2021-10/gcreplay', help='dir with gctree results on gcreplay data from which we read seqs, affinity, mutation info, and trees)')
parser.add_argument("--beast-version", default='2023-05-18-beast', help='subdir of --input-dir with beast results')
parser.add_argument("--output-version", default='test')
parser.add_argument("--max-leaf-count", type=int, default=100)
parser.add_argument("--igk-idx", type=int, default=336, help='zero-based index of first igk position in smooshed-together igh+igk sequence')
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--check-gct-kd", action='store_true')
parser.add_argument("--variant-score-fname", default='projects/gcdyn/experiments/final_variant_scores.csv')
parser.add_argument("--cgg-naive-sites-fname", default='projects/gcdyn/experiments/CGGnaive_sites.csv')
parser.add_argument("--test", action='store_true')
args = parser.parse_args()

baseoutdir = '/fh/fast/matsen_e/data/taraki-gctree-2021-10/beast-processed-data/%s' % args.output_version

rpmeta = datautils.read_gcreplay_metadata(args.gcreplay_dir)

dms_df, naive_seqs, pos_maps = get_mut_info()
if args.check_gct_kd:
    gct_kd_vals = read_gct_kd()

infokeys = ['encoded_trees', 'sstats', 'lmetafos', 'seqfos', 'dtrees', 'metafos']
all_info = {k : [] for k in infokeys}
for idir, bstdir in enumerate(glob.glob('%s/%s/beast/*-GC' % (args.input_dir, args.beast_version))):
    read_single_dir(bstdir, idir)
    if args.test and idir > 3:
        break

all_tps = sorted(set(m['time'] for m in rpmeta.values()))
for tps in all_tps:
    subfo = subset_info(tps)
    if len(subfo['encoded_trees']) > 0:
        write_output('%s/%s-trees' % (baseoutdir, tps), subfo)
write_output('%s/all-trees' % baseoutdir, all_info)
