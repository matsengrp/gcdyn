r"""Utility functions ^^^^^^^^^^^^^^^^^"""

# fmt: off

from collections import defaultdict, OrderedDict

import ete3
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform
import copy
import matplotlib as mpl
import operator
import seaborn as sns
import platform
import resource
import psutil
import os
import glob
import time
import pandas as pd
import warnings
import sys
import itertools
from Bio.Seq import Seq
import math
import string
import csv

sigmoid_params = ['xscale', 'xshift', 'yscale', 'yshift']  # ick
slp_vals = ['max_slope', 'x_max_slope', 'init_birth', 'mean_val', 'max_val'] #, 'max_diff']
affy_bins = [-15, -10, -7, -5, -3, -2, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 2.5, 3.5, 4, 5, 7]
# zoom_affy_bins = [-2, -1, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 2, 2.5]
zoom_affy_bins = [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
fitness_bins = [-1.5, -1, -0.75, -0.5, -0.2, 0, 0.1, 0.25, 0.4, 0.5, 0.75, 1, 1.5, 2] #, 5, 15]
pltlabels = {
    'affinity' : 'affinity ($ -\Delta log_{10} K_D $)',
    'init_birth' : 'naive birth rate',
    'x_max_slope' : 'x value of max slope',
    'max_slope' : 'max slope',
    'mean_val' : 'mean value',
    'max_val' : 'max value',
    'max_diff' : 'max diff',
}

def simple_fivemer_contexts(sequence: str):
    r"""Decompose a sequence into a list of its 5mer contexts.

    Args:
        sequence: A nucleotide sequence.
    """
    return tuple([sequence[(i - 2) : (i + 3)] for i in range(2, len(sequence) - 2)])


def padded_fivemer_contexts_of_paired_sequences(sequence: str, chain_2_start_idx: int):
    r"""Given a pair of sequences with two chains, split them apart at the given
    index, then generate all of the 5mer contexts for each of the subsequences,
    padding appropriately.

    Args:
        sequence: A nucleotide sequence.
        chain_2_start_idx: The index at which to split the sequence into two
    """
    chain_1_seq = "NN" + sequence[:chain_2_start_idx] + "NN"
    chain_2_seq = "NN" + sequence[chain_2_start_idx:] + "NN"
    return simple_fivemer_contexts(chain_1_seq) + simple_fivemer_contexts(chain_2_seq)


def node_contexts(node: ete3.TreeNode):
    if hasattr(node, "chain_2_start_idx"):
        return padded_fivemer_contexts_of_paired_sequences(
            node.sequence, node.chain_2_start_idx
        )
    else:
        return simple_fivemer_contexts(node.sequence)


def ltranslate(nuc_seq):
    return str(Seq(nuc_seq).translate())


def write_leaf_sequences_to_fasta(
    tree: ete3.TreeNode, file_path: str, naive: bool = None
):
    r"""Write the sequences at the leaves of a tree to a FASTA file, potentially
    including a naive sequence.

    Args:
        tree: A tree with sequences at the leaves.
        file_path: The path to the FASTA file to write.
        naive: Flag to include a naive root sequence in output.
    """
    sequence_dict = {leaf.name: leaf.sequence for leaf in tree.iter_leaves()}
    if naive is not None:
        sequence_dict["naive"] = naive
    with open(file_path, "w") as fp:
        for name, sequence in sequence_dict.items():
            fp.write(f">{name}\n")
            fp.write(f"{sequence}\n")

    return sequence_dict


def ladderize_tree(tree, attr="x", check_for_ties=False, debug=False):
    """
    Ladderizes the given tree.
    Adapts the procedure described in Voznica et. al (2022) for trees whose leaves
    may occur at the same time.

    *This is done in place!*
    Assumes that the `node.t` time attribute is ascending from root to leaf.

    First, we compute the following values for each node in the tree:
        1. The time of the leaf in the subtree at/below this node which is
           closest to present time
        2. The time of the ancestor node immediately prior to that leaf
        3. The attribute value `attr` (given in arguments) of that leaf

    Then, every node has its child subtrees reordered to sort by these values, decreasing from left to right (which corresponds to most recent and largest `attr` first).
    Values 2 and 3 are tie-breakers for value 1.

    Voznica, J., A. Zhukova, V. Boskova, E. Saulnier, F. Lemoine, M. Moslonka-Lefebvre, and O. Gascuel. “Deep Learning from Phylogenies to Uncover the Epidemiological Dynamics of Outbreaks.” Nature Communications 13, no. 1 (July 6, 2022): 3896. https://doi.org/10.1038/s41467-022-31511-0.
    """
    # ----------------------------------------------------------------------------------------
    def sortstr(tnd, vals=None):
        return ' '.join(('%6.3f'%v) for v in (sort_criteria[tnd.name] if vals is None else vals))
    # ----------------------------------------------------------------------------------------
    def check_ties(nodelist, sub_root):  # note that ties are only a problem if the occur on *non*-identical subtrees
        def kfn(n): return sort_criteria[n.name]
        svgroups = [(vls, list(gp)) for vls, gp in itertools.groupby(sorted(nodelist, key=kfn, reverse=True), key=kfn)]  # group together nodes with the same sort criteria
        len_sortd_groups = sorted(svgroups, key=lambda gp: len(gp[1]), reverse=True)  # sort by the lengths of these groups
        if len(len_sortd_groups[0][1]) > 1:  # if largest (first) group is longer than 1 (has a tie)
            print('        %s multiple nodes with same sort criteria%s:' % (color('yellow', 'warning'), ' below node %s'%sub_root.name))
            for svals, sgroup in len_sortd_groups:
                if len(sgroup) > 1:
                    print('          %s    %s' % (sortstr(None, vals=svals), ' '.join(str(n.name) for n in sgroup)))
            print(pad_lines(sub_root.get_ascii(), extra_str='              '))
    # ----------------------------------------------------------------------------------------
    def age_sum(node, tdbg=False):  # sum of ages from node to root
        if tdbg:
            print('  age sum for %s' % node.name)
        total, parent = 0, node
        while not parent.is_root():
            total += parent.t
            if tdbg:
                print('        %15s  %5.2f  %5.2f' % (parent.name, parent.t, total))
            parent = parent.up
        return total
    # ----------------------------------------------------------------------------------------
    def get_criteria(node):  # note that node.up.t would always be the same (the parent of the two nodes we're sorting), except the sort values of each internal node are the sort values of its first child (so, eventually, always a leaf)
        return [node.t, node.up.t, age_sum(node), getattr(node, attr)]
    # ----------------------------------------------------------------------------------------
    if debug:
        print('      before ladderization:')
        # print(pad_lines(tree.get_ascii(show_internal=True)))
        print_as_dtree(tree)

    # calculate sorting criteria values
    sort_criteria = defaultdict(list)
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            sort_criteria[node.name] = get_criteria(node)
        else:
            sort_criteria[node.name] = sorted(  # sort criteria for internal nodes is the criteria of their first (by sort criteria) child
                (sort_criteria[child.name] for child in node.children), reverse=True
            )[0]  # ties here won't change anything right *here*, but changing the sort criteria (by adding more values) would change this

    # sort by those criteria
    for node in tree.traverse("postorder"):
        if len(node.children) > 1:
            if check_for_ties:
                check_ties(node.children, sub_root=node)
            node.children = sorted(
                node.children,
                key=lambda n: sort_criteria[n.name],
                reverse=True,
            )
            if debug:
                print('  %-8s %d children' % (node.name+':', len(node.children)))
                for chd in node.children:
                    print('     %-8s %s' % (chd.name+':', sortstr(chd)))

    if debug:
        print('      after ladderization:')
        # print(pad_lines(tree.get_ascii(show_internal=True)))
        print_as_dtree(tree)


def random_transition_matrix(length, seed=None):
    rng = np.random.default_rng(seed)

    mat = np.abs(
        np.array([uniform.rvs(size=length, random_state=rng) for _ in range(length)])
    )

    for i in range(length):
        mat[i, i] = 0
        mat[i, :] /= sum(mat[i, :])

    return mat


def plot_responses(*responses, x_range=(-10, 10), **named_responses):
    x_array = np.linspace(*x_range)

    plt.figure()

    for response in responses:
        plt.plot(x_array, response.λ_phenotype(x_array), color="black", alpha=0.5)

    for name, response in named_responses.items():
        plt.plot(x_array, response.λ_phenotype(x_array), label=name)

    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.xlim(*x_range)

    if named_responses:
        plt.legend()

    plt.show()


# ----------------------------------------------------------------------------------------
# bash color codes
ansi_color_table = OrderedDict(
    (
        # 'head' : '95'  # not sure wtf this was?
        ("end", 0),
        ("bold", 1),
        ("reverse_video", 7),
        ("grey", 90),
        ("red", 91),
        ("green", 92),
        ("yellow", 93),
        ("blue", 94),
        ("purple", 95),
        ("grey_bkg", 100),
        ("red_bkg", 41),
        ("green_bkg", 42),
        ("yellow_bkg", 43),
        ("blue_bkg", 44),
        ("light_blue_bkg", 104),
        ("purple_bkg", 45),
    )
)
Colors = {c: "\033[%sm" % i for c, i in ansi_color_table.items()}


# ----------------------------------------------------------------------------------------
def color(col, seq, width=None, padside="left"):
    return_str = [seq]
    if col is not None:
        return_str = [Colors[col]] + return_str + [Colors["end"]]
    if width is not None:  # make sure final string prints to correct width
        n_spaces = max(
            0, width - len(seq)
        )  # if specified <width> is greater than uncolored length of <seq>, pad with spaces so that when the colors show up properly the colored sequences prints with width <width>
        if padside == "left":
            return_str.insert(0, n_spaces * " ")
        elif padside == "right":
            return_str.insert(len(return_str), n_spaces * " ")
        else:
            assert False
    return "".join(return_str)


# ----------------------------------------------------------------------------------------
def color_mutants(ref_seq, qseq, amino_acid=False):  # crappy version of fcn in partis utils
    def getcol(r, q):
        if q in ambig_bases or r in ambig_bases:
            return 'blue'
        elif q == r:
            return None
        else:
            return "red"
    assert len(ref_seq) == len(qseq)
    ambig_bases = 'X' if amino_acid else 'N-'
    return "".join([color(getcol(r, q), q) for r, q in zip(ref_seq, qseq)])


# ----------------------------------------------------------------------------------------
def non_none(vlist):  # return the first non-None value in vlist (there are many, many places where i could go back and use this) [this avoids hard-to-read if/else statements that require writing the first val twice]
    for val in vlist:
        if val is not None:
            return val
    raise Exception('utils.non_none() called with all-None vlist: %s' % vlist)

# ----------------------------------------------------------------------------------------
def isclose(num1, num2, eps=1e-8, debug=False, fail=False, warn=False):
    """Return true if num1 and num2 are closer to each other than eps (numpy version is super slow."""
    if abs(num1 - num2) < eps:
        return True
    dbgstr = "%snumbers %.10f and %.10f not closer than eps %.10f (abs diff %.10f)" % (
        color("yellow", "warning ") if warn else "",
        num1,
        num2,
        eps,
        abs(num1 - num2),
    )
    if fail:
        raise Exception(dbgstr)
    if debug or warn:
        print("    " + dbgstr)
    return False


# ----------------------------------------------------------------------------------------
# The following functions (mostly with <arglist> in the name) are for manipulating lists of command line arguments
# (here called "clist", e.g. from sys.argv) to, for instance, allow a script to modify its own arguments for use in running
# subprocesses of itself with similar command lines. They've been copied from partis/utils.py.


# return true if <argstr> is in <clist>
def is_in_arglist(
    clist, argstr
):  # accounts for argparse unique/partial matches (update: we no longer allow partial matches)
    return len(arglist_imatches(clist, argstr)) > 0


# return list of indices matching <argstr> in <clist>
def arglist_imatches(clist, argstr):
    assert (
        argstr[:2] == "--"
    )  # this is necessary since the matching assumes that argparse has ok'd the uniqueness of an abbreviated argument UPDATE now we've disable argparse prefix matching, but whatever
    return [
        i for i, c in enumerate(clist) if argstr == c
    ]  # NOTE do *not* try to go back to matching just the start of the argument, in order to make that work you'd need to have access to the whole list of potential arguments in bin/partis, and you'd probably screw it up anyway


# return index of <argstr> in <clist>
def arglist_index(clist, argstr):
    imatches = arglist_imatches(clist, argstr)
    if len(imatches) == 0:
        raise Exception("'%s' not found in cmd: %s" % (argstr, " ".join(clist)))
    if len(imatches) > 1:
        raise Exception("too many matches")
    return imatches[0]


# replace the argument to <argstr> in <clist> with <replace_with>, or if <argstr> isn't there add it. If we need to add it and <insert_after> is set, add it after <insert_after>
def replace_in_arglist(clist, argstr, replace_with, insert_after=None, has_arg=False):
    if clist.count(None) > 0:
        raise Exception("None type value in clist %s" % clist)
    if not is_in_arglist(clist, argstr):
        if insert_after is None or insert_after not in clist:  # just append it
            clist.append(argstr)
            clist.append(replace_with)
        else:  # insert after the arg <insert_after>
            insert_in_arglist(
                clist, [argstr, replace_with], insert_after, has_arg=has_arg
            )
    else:
        clist[arglist_index(clist, argstr) + 1] = replace_with


# insert list <new_arg_strs> after <argstr> (unless <before> is set),  Use <has_arg> if <argstr> has an argument after which the insertion should occur
def insert_in_arglist(
    clist, new_arg_strs, argstr, has_arg=False, before=False
):  # set <argstr> to None to put it at end (yeah it probably should've been a kwarg)
    i_insert = len(clist)
    if argstr is not None:
        i_insert = clist.index(argstr) + (2 if has_arg else 1)
    clist[i_insert:i_insert] = new_arg_strs


def remove_from_arglist(clist, argstr, has_arg=False):
    if clist.count(None) > 0:
        raise Exception("None type value in clist %s" % clist)
    imatches = arglist_imatches(clist, argstr)
    if len(imatches) == 0:
        return
    if len(imatches) > 1:
        assert False  # not copying this fcn from partis (shouldn't get here atm, but leaving it commented to provide context in case it does get triggered)
        # imatches = reduce_imatches(imatches, clist, argstr)
    iloc = imatches[0]
    # if clist[iloc] != argstr:
    #     print '  %s removing abbreviation \'%s\' from sys.argv rather than \'%s\'' % (color('yellow', 'warning'), clist[iloc], argstr)
    if has_arg:
        clist.pop(iloc + 1)
    clist.pop(iloc)
    return clist  # NOTE usually we don't use the return value (just modify it in memory), but at least in one place we're calling with a just-created list so it's nice to be able to use the return value


# ----------------------------------------------------------------------------------------
def mpl_init(fsize=20, label_fsize=15):
    sns.set_style("ticks")
    # sns.set_palette("viridis", 8)
    mpl.rcParams.update(
        {
            "font.size": fsize,
            "axes.titlesize": fsize,
            "axes.labelsize": fsize,
            "xtick.labelsize": label_fsize,
            "ytick.labelsize": label_fsize,  # NOTE this gets (maybe always?) overriden by xticklabelsize/yticklabelsize in mpl_finis()
            "legend.fontsize": fsize,
            "font.family": "sans-serif",
            "font.sans-serif": ["Lato", "DejaVu Sans"],
            "font.weight": 600,
            "axes.labelweight": 600,
            "axes.titleweight": 600,
            "figure.autolayout": True,
        }
    )


# ----------------------------------------------------------------------------------------
# plot the two column <xkey> and <ykey> in <tdf> vs each other (pass in all_xvals separately for bounds/bins stuff since we want to have the same range for all [train/test/validation] dfs)
def sns_xy_plot(ptype, smpl, tdf, xkey, ykey, all_xvals=None, true_x_eq_y=False, discrete=False, n_bins=15, bin_edges=None, leave_ticks=False, xtra_text=None, emph_vals=None, n_max_points=None, bad_cdiff_val=-1.5):
    # ----------------------------------------------------------------------------------------
    def tcolor(def_val):
        return 'darkgreen' if smpl == 'true' else ('red' if smpl=='valid' else def_val)
    # ----------------------------------------------------------------------------------------
    def addtext(txt, xtra_text):
        if xtra_text is None:
            xtra_text = []
        xtra_text.append(txt)
        return xtra_text
    # ----------------------------------------------------------------------------------------
    if all_xvals is None:
        all_xvals = tdf[xkey]
    assert len(tdf[xkey]) == len(tdf[ykey])
    i_none = [i for (i, x, y, d) in zip(tdf.index, tdf[xkey], tdf[ykey], tdf['curve-diff-predicted']) if any(v is None or math.isnan(v) for v in [x, y, d])]
    if len(i_none) > 0:
        xtra_text = addtext('%d non-None values (of %d)' % (len(tdf[xkey]) - len(i_none), len(tdf[xkey])), xtra_text)
        tdf = tdf.loc[[i for i in tdf.index if i not in i_none]]
    if len(tdf) > n_max_points and ptype == 'scatter':
        xtra_text = addtext('%s%d/%d points' % ('%s: '%smpl, n_max_points, len(tdf)), xtra_text)
        tdf = tdf.sample(n=n_max_points)
    if ptype == 'scatter':
        if discrete:
            ax = sns.swarmplot(tdf, x=xkey, y=ykey, size=4, alpha=0.6, order=sorted(set(all_xvals)))  # ax.set(title=smpl)
        else:
            ax = sns.scatterplot(tdf, x=xkey, y=ykey, alpha=0.6, color=tcolor(None))
            bad_rows = tdf[tdf['curve-diff-predicted']<bad_cdiff_val]
            # if len(bad_rows) > 0:
            #     print('  %s %s' % (xkey, ykey))
            #     print('    %d / %d bad rows (cdiff < %.2f)' % (len(bad_rows), len(tdf), bad_cdiff_val))
            ax = sns.scatterplot(bad_rows, x=xkey, y=ykey, alpha=0.6, color='red', edgecolor='black', facecolor=None, s=50)
            if true_x_eq_y: # and smpl in ['train', 'infer']:  # 'train' and 'valid' go on the same plot, but we don't want to plot the dashed line twice
                plt.plot([0.95 * min(all_xvals), 1.05 * max(all_xvals)], [0.95 * min(all_xvals), 1.05 * max(all_xvals)], color="darkgreen", linestyle="--", linewidth=3, alpha=0.7)
        if emph_vals is not None:
            ax.scatter([emph_vals[0][0]], [emph_vals[0][1]], color='red', alpha=0.45, s=250)
    elif ptype == 'box':
        if discrete:
            bkey, order = xkey, sorted(set(all_xvals))
        else:
            if bin_edges is None:
                dx = (max(all_xvals) - min(all_xvals)) / float(n_bins)
                bin_edges = np.arange(min(all_xvals), max(all_xvals) + dx, dx)
            tdf['x_bins'] = pd.cut(tdf[xkey], bins=bin_edges) #n_bins)
            bkey, order = 'x_bins', None
        boxprops = {"facecolor": "None", 'edgecolor': tcolor('black'), 'linewidth' : 5 if smpl=='valid' else 2}
        ax = sns.boxplot(x=bkey, y=ykey, data=tdf, order=order, boxprops=boxprops, whiskerprops={'color' : tcolor('black')}, showfliers=False) #, ax=ax)
        if not leave_ticks:  # if we're calling this fcn twice (i.e. plotting validation set), this tick/true line stuff is already there (and will crash if called twice)
            xtls = []
            for xv, xvl in zip(ax.get_xticks(), ax.get_xticklabels()):  # plot a horizontal dashed line at y=x (i.e. correct) for each bin and get labels
                if discrete:
                    tyv = float(xvl._text)
                else:
                    lo, hi = [float(s.lstrip('(').rstrip(']')) for s in xvl._text.split(', ')]
                    tyv = np.mean([lo, hi])  # true/y val
                    xtls.append('%.1f-%.1f'%(lo, hi))
                if true_x_eq_y:
                    plt.plot([xv - 0.5, xv + 0.5], [tyv, tyv], color="darkgreen", linestyle="--", linewidth=3, alpha=0.7)
            if not discrete:
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(xtls, rotation=90, ha='right')
    else:
        assert False
    if xtra_text is not None:
        for itxt, ttxt in enumerate(xtra_text):
            ypos = 0.75 if smpl=='valid' else 0.85
            plt.text(0.05, ypos - 0.1 * itxt, ttxt, transform=ax.transAxes, color='red' if smpl=='valid' else None)
    return ax

# ----------------------------------------------------------------------------------------
# plot scatter + box/whisker plot comparing true and predicted values for deep learning inference
# NOTE leaving some commented code that makes plots we've been using recently, since we're not sure which plots we'll end up wanting in the end (and what's here is very unlikely to stay for very long)
def make_dl_plots(model_type, prdfs, seqmeta, params_to_predict, outdir, is_simu=False, data_val=0, validation_split=0, xtra_txt=None, fsize=20, label_fsize=15, trivial_encoding=False, nonsense_affy_val=-99, force_many_plot=False):
    quick = False  # turn this on to speed things up by skipping some plots
    # ----------------------------------------------------------------------------------------
    def add_fn(fn, n_per_row=4, force_new_row=False, fns=None):
        if fns is None:
            fns = fnames  # from parent scope, which is kind of ugly, but keeps things more concise
        if force_new_row or len(fns) == 0 or len(fns[-1]) >= n_per_row:
            fns.append([])
        fns[-1].append(fn)
    # ----------------------------------------------------------------------------------------
    def plot_responses(smpl, n_max_plots=2 if quick else 20, n_max_diffs=100 if quick else 1000, default_xbounds=None):
        # ----------------------------------------------------------------------------------------
        def plot_true_pred_pair(true_resp, pred_resp, affy_vals, diff_vals, plotname, titlestr=None, xbounds=None):
            fn = plot_many_curves(outdir+'/'+smpl, plotname, [{'birth-response' : r} for r in [true_resp, pred_resp]], titlestr=titlestr,
                                  affy_vals=affy_vals, colors=['#006600', '#990012'], add_true_pred_text=True, diff_vals=diff_vals, xbounds=xbounds)
            add_fn(fn)
        # ----------------------------------------------------------------------------------------
        def get_nn_curve_loss(resp_1, resp_2):
            from gcdyn.nn import base_curve_loss
            import tensorflow as tf
            def split_resp(r): return [float(v) for v in [r.xscale, r.xshift, r.yscale, r.yshift]]
            return base_curve_loss(tf.constant([split_resp(resp_1)]), tf.constant([split_resp(resp_2)])).numpy()
        # ----------------------------------------------------------------------------------------
        def init_affy(lmdict, tree_index):
            affy_vals = [float(m['affinity']) for m in lmdict[tree_index] if float(m['affinity'])!=nonsense_affy_val]
            affy_xbds = [mfn([a for a in affy_vals if a!=nonsense_affy_val]) for mfn in [min, max]]  # these bounds go way too far to the left, which is quite different to the loss function bounds (since we don't know the affinites in that function), plus very negative affinities aren't pretty uninformative
            affy_xbds[0] = default_xbounds[0]  # this is ugly, but we don't really care about super low affinity values (although they also don't change the cdiff loss)
            return affy_vals, affy_xbds
        # ----------------------------------------------------------------------------------------
        def truncate_phist(htmp, affy_xbds):  # set prediction to zero if there were no affinity values in this bin (empty bins are then ignored when getting mean hist and plotting)
            txmin, txmax = affy_xbds
            for ibin in htmp.ibiniter(include_overflows=False):
                if txmin > htmp.low_edges[ibin+1] or txmax < htmp.low_edges[ibin]:
                    htmp.set_ibin(ibin, 0, 0)
        # ----------------------------------------------------------------------------------------
        def getresp(ptype, plist, irow):
            pdict = {p : prdfs[smpl]['%s-%s'%(p, ptype)][irow] for p in plist}
            if 'y_ceil' in pdict and pdict['y_ceil'] is not None and math.isnan(pdict['y_ceil']):  # for some reason pandas reads an empty column as nan instead of None
                pdict['y_ceil'] = None
            resp_type = SigmoidCeilingResponse if ptype=='truth' and 'x_ceil_start' in pdict else SigmoidResponse
            return resp_type(**pdict)
        # ----------------------------------------------------------------------------------------
        if model_type == 'per-cell':
            raise Exception('needs updating (see commented block below)')
        if default_xbounds is None:
            default_xbounds = [-2.5, 3]
        default_ybounds = [-0.1, 15]
        if not os.path.exists(outdir+'/'+smpl):
            os.makedirs(outdir+'/'+smpl)
        from gcdyn.poisson import SigmoidResponse, SigmoidCeilingResponse, LinearResponse
        median_pfo = None
        inf_pfo_list, true_pfo_list, all_afvals = [], [], []
        curve_diffs = {smpl : [], 'validation' : []}
        n_pred_rows = len(prdfs[smpl].index)  # N cells for per-cell, N trees for sigmoid
        n_tree_preds = len(set(prdfs[smpl]['tree-index']))
        n_simu_plots = min(n_max_plots, n_tree_preds)
        lmdict = defaultdict(list)  # map from tree index to list of seq meta for that tree
        validation_indices = None
        if smpl == "train" and validation_split != 0:  # NOTE this obviously depends on keras continuing to do validation splits this way
            validation_indices = [i for i in range(n_tree_preds - int(validation_split * n_tree_preds), n_tree_preds)]
        for mfo in seqmeta:
            lmdict[int(mfo['tree-index'])].append(mfo)
            n_skipped_diffs = {s : 0 for s in curve_diffs}
        if model_type == 'sigmoid':
            pstr, rkey, nsteps, is_hist = 'response', 'birth-response', 40, False
        elif model_type == 'per-bin':
            import gcdyn.encode as encode
            pstr, rkey, nsteps, is_hist = 'fitness-bins', 'birth-hist', None, True
        else:
            assert False  # per-cell needs (minimal) updating
        for irow in range(n_tree_preds):
            tree_index = int(prdfs[smpl]['tree-index'][irow])
            affy_vals, affy_xbds = init_affy(lmdict, tree_index)
            is_valid = validation_indices is not None and tree_index in validation_indices
            smplstr = 'validation' if is_valid else smpl
            if model_type == 'sigmoid':
                pred_resp = getresp('predicted', sigmoid_params, irow)
                pred_pfo = {'birth-response' : pred_resp, 'xbounds' : default_xbounds} #affy_xbds
            elif model_type == 'per-bin':
                pbvals = [prdfs[smpl]['fitness-bins-predicted-ival-%d'%ival][irow] for ival in range(len(zoom_affy_bins)-1)]  # could also get the bin truth values if we wanted them
                pred_pfo = {'birth-hist' : encode.decode_fitness_bins(pbvals, zoom_affy_bins)}
            else:
                assert False
            if model_type == 'per-bin':
                add_slope_vals(pred_pfo[rkey], default_xbounds, pred_pfo, is_hist=is_hist)  # these are really slow, so turning off
            # truncate_phist(pred_pfo['birth-hist'], affy_xbds)  # would of course only turn this for per-bin
            if is_simu:
                titlestr = '%s: response index %d / %d' % (smplstr, irow, n_tree_preds)
                true_pfo = {'birth-response' : getresp('truth', (sigmoid_params + ['x_ceil_start', 'y_ceil']) if 'x_ceil_start-truth' in prdfs[smpl] else sigmoid_params, irow)}
                if model_type == 'per-bin':
                    add_slope_vals(true_pfo['birth-response'], default_xbounds, true_pfo)  # these are really slow, so turning off
                true_pfo_list.append(true_pfo)
                cdiff = None
                if len(curve_diffs[smplstr]) < n_max_diffs:
                    cdiff = resp_fcn_diff(true_pfo['birth-response'], pred_pfo[rkey], default_xbounds, resp2_is_hist=is_hist, nsteps=nsteps)
                    pred_pfo['curve-diff'] = cdiff  # I don't like putting the cdiff both in here and in curve_diffs, but the latter splits apart train/valid but inf_pfo_list doesn't (and they subsequently get used in different ways)
                    curve_diffs[smplstr].append(cdiff)  # NOTE len(curve_diffs[smplstr]) is *not* the same as irow since e.g. train and validation come from same sample
                    # ldiff = get_curve_loss(true_pfo['birth-response'], pred_resp)  # uncomment (also below) to print also the diff from the curve loss fcn (it should be the same, modulo different xbounds and nsteps
                else:
                    n_skipped_diffs[smplstr] += 1
                if irow < n_simu_plots: # or pred_resp.yscale>55 or (cdiff is not None and cdiff > 0.75):
                    if model_type=='sigmoid':
                        pfl, phists, colors = [true_pfo, pred_pfo], None, ['#006600', '#1f77b4']
                    else:
                        pfl, phists, colors = [true_pfo], [pred_pfo], ['#006600']
                    fn = plot_many_curves(outdir+'/'+smpl, 'true-vs-inf-%s-%d'%(pstr, irow), pfl, pred_hists=phists,
                                          titlestr=titlestr, xbounds=default_xbounds, affy_vals=affy_vals, colors=colors,
                                          diff_vals=[{'diff' : cdiff}] if cdiff is not None else None, param_text_pfos=[true_pfo, pred_pfo])  # , 'xbounds' : default_xbounds
                    add_fn(fn)
            else:
                titlestr = '%s (%d / %d)' % (prdfs[smpl]['gcids'][irow] if 'gcids' in prdfs[smpl] else smplstr, irow, n_tree_preds)
                pfl, phists, colors = ([pred_pfo], None, ['#1f77b4']) if model_type=='sigmoid' else ([], [pred_pfo], None)
                fn = plot_many_curves(outdir+'/'+smpl, 'predicted-%s-%d'%(pstr, irow), pfl, pred_hists=phists, param_text_pfos=[None, pred_pfo],
                                      titlestr=titlestr, affy_vals=affy_vals, xbounds=default_xbounds, ybounds=default_ybounds, colors=colors)
                add_fn(fn, fns=single_curve_fns)
                if irow < n_max_plots:
                    add_fn(fn)
            inf_pfo_list.append(pred_pfo)
            all_afvals += affy_vals
        for tk in slp_vals + ['curve-diff']:
            prdfs[smpl]['%s-predicted'%tk] = [p.get(tk) for p in inf_pfo_list]
        if is_simu:
            for tk in slp_vals:
                prdfs[smpl]['%s-truth'%tk] = [p.get(tk) for p in true_pfo_list]
        if force_many_plot and model_type=='per-bin':
            print('    %s --force-many-plot needs to be checked for per-bin model' % wrnstr())
        if force_many_plot or not is_simu or all(p['birth-response']==true_pfo_list[0]['birth-response'] for p in true_pfo_list):  # all true responses equal
            # median_pfo = {'birth-hist' : make_mean_hist([p['birth-hist'] for p in inf_pfo_list], ignore_empty_bins=True, percentile_err=True)}
            median_pfo = copy.copy(get_median_curve(inf_pfo_list, default_xbounds, nsteps=nsteps, is_hist=is_hist))  # copy is because it returns an element from the list
            add_slope_vals(median_pfo[rkey], default_xbounds, median_pfo, is_hist=is_hist)

            if model_type == 'sigmoid':
                # plt_pfos, colors, alphas, phists = inf_pfo_list, ['#1f77b4' for _ in inf_pfo_list], [0.05 for _ in inf_pfo_list], None  # this line gives you the old-style plot-all-curves-with-low-alpha
                plt_pfos, colors, alphas, phists = [], [], [], None
            else:
                # plt_pfos, colors, alphas, phists = [], (['#006600'] if is_simu else None), ([] if is_simu else None), inf_pfo_list  # this line gives you the old-style plot-all-curves-with-low-alpha
                plt_pfos, colors, alphas, phists = [], (['#006600'] if is_simu else None), ([] if is_simu else None), None
            if is_simu:
                tpfo = copy.copy(true_pfo_list[0])
                if force_many_plot:
                    tpfo = copy.copy(get_median_curve(true_pfo_list, default_xbounds))
                    plt_pfos.append(tpfo)
                    colors.append('green')
                    alphas.append(0.5)
                add_slope_vals(tpfo['birth-response'], default_xbounds, tpfo)  # re-get slope vals with default bounds
                cdiff = resp_fcn_diff(tpfo['birth-response'], median_pfo[rkey], default_xbounds, resp2_is_hist=is_hist, nsteps=nsteps)
                plt_pfos += true_pfo_list if force_many_plot else [tpfo]
                colors += ['green' for _ in range(len(plt_pfos)-len(colors))]
                alphas += [(0.05 if force_many_plot else 0.5) for _ in range(len(plt_pfos)-len(alphas))]
            titlestr = '%s: %d responses' % (smpl, n_tree_preds)  # '%s: %d / %d responses' % (smpl, len(inf_pfo_list), n_tree_preds)
            fn = plot_many_curves(outdir+'/'+smpl, '%s-all-%s'%(smpl, pstr), plt_pfos, pred_hists=phists, ci_pfos=inf_pfo_list,
                                  titlestr=titlestr, median_pfo=median_pfo, affy_vals=all_afvals, xbounds=default_xbounds, ybounds=default_ybounds, diff_vals=[cdiff] if is_simu else None,
                                  colors=colors, alphas=alphas, param_text_pfos=[tpfo, 'median'] if is_simu else [None, 'median'])
            add_fn(fn, force_new_row=True)
        if is_simu:
            fn = plot_all_diffs(outdir+'/'+smpl, 'curve-diffs', curve_diffs, n_skipped_diffs=n_skipped_diffs)
            add_fn(fn)
        # NOTE I just combined the sigmoid+per-cell blocks (above), and still need to integrate this or put in a separate block, but don't want to bother now since I'm not sure I'll ever need it
        # elif model_type == 'per-cell':
        #     all_xvals, all_yvals, all_true_responses = [], [], []
        #     irow = 0  # row in file (per-cell, unlike previous block)
        #     itree = 0  # itree is zero-based index/count of tree in this file, whereas tree_index is index in original simulation sequence (e.g. if this file starts from 10th tree, itree starts at 0 but tree_index starts at 9)
        #     while itree < n_max_plots:
        #         tree_index = int(prdfs[smpl]['tree-index'][irow])
        #         if trivial_encoding:
        #             true_resp = LinearResponse()
        #             xbounds = None
        #         else:
        #             if is_simu:
        #                 pdict = {p : prdfs[smpl]['%s-truth'%p][irow] for p in sigmoid_params}
        #                 true_resp = SigmoidResponse(**pdict)
        #             xbounds = [mfn([float(m['affinity']) for m in lmdict[tree_index]]) for mfn in [min, max]]
        #         tpfl = [{'birth-response' : true_resp, 'xbounds' : xbounds}] if is_simu else []  # short for true_pfo_list, but this one we only use within this block
        #         dfdata = {'phenotype' : [], 'fitness-predicted' : []}
        #         while irow < n_pred_rows and int(prdfs[smpl]['tree-index'][irow]) == tree_index:  # until next tree
        #             for tk in dfdata:
        #                 dfdata[tk].append(prdfs[smpl][tk][irow])
        #             irow += 1
        #         smpstr = smpl if validation_indices is None or tree_index not in validation_indices else 'validation'
        #         fn = plot_many_curves(outdir+'/'+smpl, 'true-vs-inf-response-%d'%itree, tpfl, titlestr='%s: response index %d / %d' % (smpstr, itree, n_tree_preds),
        #                               colors=['#006600'], pred_xvals=dfdata['phenotype'], pred_yvals=dfdata['fitness-predicted']) #, xbounds=xbounds)
        #         assert len(dfdata['phenotype']) == len(dfdata['fitness-predicted'])
        #         all_xvals += dfdata['phenotype']
        #         all_yvals += dfdata['fitness-predicted']
        #         if is_simu:
        #             all_true_responses.append(tpfl[0])
        #         add_fn(fn)
        #         itree += 1
        #     fn = plot_many_curves(outdir+'/'+smpl, 'true-vs-inf-response', tpfl, titlestr='%s: %d / %d responses' % (smpl, len(pfo_list), n_tree_preds),
        #                           pred_xvals=all_xvals, pred_yvals=all_yvals, xbins=zoom_affy_bins, xbounds=default_xbounds, colors=['green' for _ in tpfl])
        #     add_fn(fn)

        return median_pfo
    # ----------------------------------------------------------------------------------------
    def plot_param_or_pair(ptype, param1, smpl, xkey=None, ykey=None, param2=None, vtypes=['truth', 'predicted'], median_pfo=None, n_max_scatter_points=25 if quick else 750):
        # # ----------------------------------------------------------------------------------------
        # def get_mae_text(smpl, tdf):
        #     if not is_simu:
        #         return None
        #     mae = np.mean([
        #         abs(pval - tval)
        #         for tval, pval in zip(tdf[xkey], tdf['%s-predicted'%param1])
        #     ])
        #     return '%s mae: %.4f' % (smpl, mae)
        # ----------------------------------------------------------------------------------------
        def get_pval(pfo, pname):
            if pname in pfo:
                return pfo[pname]
            else:
                tresp = pfo['birth-response'] if 'birth-response' in pfo else pfo['birth-hist']  # well, second option is a hist not a response
                return getattr(tresp, pname)
        # ----------------------------------------------------------------------------------------
        plt.clf()
        all_df = prdfs[smpl]
        if ptype == 'scatter' and len(all_df) > n_max_scatter_points:
            if param2 is None:  # don't make true/inferred plots with lots of points (the box plots are sufficient)
                return
        xlabel, ylabel = xkey, ykey
        emph_vals = None
        if xkey is None or ykey is None:
            assert xkey is None and ykey is None
            if param2 is None:  # plotting true vs inferred values (if data, "true" is a single arbitrary value)
                xkstr = "truth" if is_simu else "data"
                xlabel, ylabel = xkstr.replace("truth", "true"), "predicted value"
                xkey, ykey = ["%s-%s" % (param1, v) for v in [xkstr, "predicted"]]  # should really switch to vtypes, but data stuff complicates that
                if not is_simu:  # set x value for data (since there's no truth, we pick an arbitrary x value at which to display the points)
                    all_df[xkey] = [data_val for _ in all_df['%s-predicted'%param1]]
                if median_pfo is not None:
                    emph_vals = [[data_val, get_pval(median_pfo, param1)]]
            else:  # plotting two different variables vs each other
                xkey, ykey = ["%s-%s" % (p, v) for p, v in zip([param1, param2], vtypes)]
                xlabel, ylabel = ['%s %s' % (v.replace('truth', 'true'), pltlabels.get(p, p)) for v, p in zip(vtypes, [param1, param2])]
                if median_pfo is not None:
                    emph_vals = [[get_pval(median_pfo, p) for p in [param1, param2]]]
        discrete = param2 is None and len(set(all_df[xkey])) < 15  # if simulation has discrete parameter values (well, mostly turned off now i think)
        bin_edges = fitness_bins if param1=='fitness' else None
        plt_df = all_df.copy()
        if smpl == "train" and validation_split != 0:
            plt_df = all_df.copy()[: len(all_df) - int(validation_split * len(all_df))]  # NOTE this obviously depends on keras continuing to do validation splits this way
            vld_df = all_df.copy()[len(all_df) - int(validation_split * len(all_df)) :]
            ax1 = sns_xy_plot(ptype, 'valid', vld_df, xkey, ykey, all_xvals=all_df[xkey], discrete=discrete, bin_edges=bin_edges, leave_ticks=True, n_max_points=n_max_scatter_points)  # well, leave ticks *and* don't plot true dashed line
        ax2 = sns_xy_plot(ptype, smpl, plt_df, xkey, ykey, all_xvals=all_df[xkey], true_x_eq_y=xkey.replace('-truth', '')==ykey.replace('-predicted', ''), discrete=discrete, bin_edges=bin_edges, emph_vals=emph_vals, n_max_points=n_max_scatter_points)
        if param2 is None:
            plt.xlabel("%s value" % xlabel)
            plt.ylabel(ylabel)
            titlestr = "%s%s" % (param1, '' if smpl=='infer' else ' '+smpl)
        else:
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            titlestr = '' if smpl=='infer' else smpl
        if xtra_txt is not None:
            titlestr += xtra_txt
        plt.title(titlestr, fontweight="bold", fontsize=20)  # if len(title) < 25 else 15)
        fn = "%s/%s%s-%s-%s-hist.svg" % (outdir, param1, '' if param2 is None else '-vs-%s'%param2, smpl, ptype)
        plt.savefig(fn)
        add_fn(fn)
    # ----------------------------------------------------------------------------------------
    mpl_init()
    mpl.rcParams["mathtext.default"] = 'regular'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fnames, single_curve_fns = [], []
    median_pfos = {}
    for smpl in sorted(prdfs, reverse=True):
        fnames.append([])
        median_pfos[smpl] = plot_responses(smpl)
        if model_type == 'sigmoid' and median_pfos[smpl] is not None:
            with open('%s/median-curve-%s.csv' % (outdir, smpl), 'w') as mfile:  # this is trying to match the train/test/infer csvs, which are written with the df to_csv fcn
                writer = csv.DictWriter(mfile, ['%s-predicted'%p for p in sigmoid_params])
                writer.writeheader()
                writer.writerow({'%s-predicted'%p : getattr(median_pfos[smpl]['birth-response'], p) for p in sigmoid_params})
    plt_params = sigmoid_params if model_type=='sigmoid' else slp_vals
    for ptype in ['scatter', 'box']:
        for smpl in sorted(prdfs, reverse=True):
            if model_type in ['sigmoid', 'per-bin']:
                fnames.append([])
                if is_simu:
                    for tpm in plt_params:
                        # plot_param_or_pair(ptype, tpm, smpl, median_pfo=median_pfos[smpl])  # true vs inferred params
                        plot_param_or_pair(ptype, tpm, smpl, param2='curve-diff', vtypes=['truth', 'predicted'])  # true params vs loss function   NOTE per-bin uses mse, so this isn't actually its loss function
            # if model_type == 'per-cell':
            #     fnames.append([])
            #     plot_param_or_pair(ptype, 'fitness', smpl)
    if model_type == 'sigmoid':  # 2d param1 vs param2 plots
        for smpl in sorted(prdfs, reverse=True):
            fnames.append([])
            for p1, p2 in itertools.combinations(plt_params, 2):
                plot_param_or_pair('scatter', p1, smpl, param2=p2, median_pfo=median_pfos[smpl], vtypes=['truth', 'truth'])
    make_html(outdir, fnames=fnames, extra_links=[('single curves', '%s/single-curves.html'%os.path.basename(outdir)), ])
    make_html(outdir, fnames=single_curve_fns, htmlfname='%s/single-curves.html'%outdir)

# ----------------------------------------------------------------------------------------
def plot_n_vs_time(plotdir, tree, max_time, itrial):
    # ----------------------------------------------------------------------------------------
    def make_n_vs_time_plot(ndata):
        fig, ax = plt.subplots()
        sns.lineplot(x=ndata['time'], y=ndata['n-nodes'], linewidth=5, alpha=0.5)
        ax.set(xlabel='time', ylabel='N nodes')
        plt.ylim(0, ax.get_ylim()[1])
        fn = "%s/n-vs-time-tree-%d.svg" % (plotdir, itrial)
        plt.savefig(fn)
        fnlist.append(fn)
    # ----------------------------------------------------------------------------------------
    def make_affy_slice_plot(tdata):
        # ----------------------------------------------------------------------------------------
        def label_fn(x, color, label):
            ax = plt.gca()  # get current axis
            ax.text(0, 0.2, label, color="black", fontsize=13, ha="left", va="center", transform=ax.transAxes)
        # ----------------------------------------------------------------------------------------
        warnings.simplefilter("ignore")  # i don't know why it has to warn me that it's clearing the fig/ax I'm passing in, and I don't know how else to stop it
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), "axes.linewidth": 2})
        palette = None  # sns.color_palette("Set2", 12)
        fgrid = sns.FacetGrid(pd.DataFrame(tdata), palette=palette, row="time", hue="time", aspect=8, height=1.2)  # create a grid with a row for each 'time'
        fgrid.map_dataframe(sns.kdeplot, x="affinity", fill=True, alpha=0.6)  # filled, smoothed hists
        fgrid.map_dataframe(sns.kdeplot, x="affinity", color="black")  # black outlines
        fgrid.map(label_fn, "time")  # iterate grid to plot labels
        # fgrid.fig.subplots_adjust(hspace=-0.7)  # adjust subplots to create overlap
        fgrid.set_titles("")  # remove subplot titles
        fgrid.set(yticks=[], xlabel="affinity", ylabel="time")  # remove yticks and set xlabel
        fgrid.despine(left=True)
        plt.suptitle("affinity vs time (tree %d)" % itrial, y=0.98)
        fn = "%s/phenotype-slices-tree-%d.svg" % (plotdir, itrial)
        plt.savefig(fn)
        fnlist.append(fn)
    # ----------------------------------------------------------------------------------------
    def get_data(dtype, n_slices):
        dt = round(max_time / float(n_slices))
        assert dtype in ['n-nodes', 'affinity']
        tdata = {'time': [], dtype: []}
        for stime in list(range(dt, max_time, dt)) + [max_time]:
            tslice = tree.slice(stime)
            if dtype == 'n-nodes':
                tdata['time'].append(stime)
                tdata['n-nodes'].append(len(tslice))
            elif dtype == 'affinity':
                for aval in tslice:
                    tdata["time"].append(stime)
                    tdata["affinity"].append(aval)
            else:
                assert False
        return tdata
    # ----------------------------------------------------------------------------------------
    mpl_init()
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    fnlist = []
    make_n_vs_time_plot(get_data('n-nodes', max_time))
    # old slice plots:
    # ugh i give up, this plot is ugly and not that useful
    # with warnings.catch_warnings():
    #     make_affy_slice_plot(get_data('affinity', '5'))

    return fnlist

# ----------------------------------------------------------------------------------------
def addfn(fnames, fn, n_columns=4):
    if len(fnames[-1]) >= n_columns:
        fnames.append([])
    fnames[-1].append(fn)

# ----------------------------------------------------------------------------------------
def plot_chosen_params(plotdir, param_counters, pbounds, n_bins=15, fnames=None):
    # ----------------------------------------------------------------------------------------
    def plot_param(pname):
        plt.clf()
        fig, ax = plt.subplots()
        xmin, xmax = [mfn(param_counters[pname]) for mfn in [min, max]]
        if pname in pbounds and pbounds[pname] is not None:
            for pbd in pbounds[pname]:
                ax.plot(
                    [pbd, pbd],
                    [0, 0.9 * ax.get_ylim()[1]],
                    color="red",
                    linestyle="--",
                    linewidth=3,
                )
            xmin, xmax = [mfn(v, b) for mfn, v, b in zip([min, max], [xmin, xmax], pbounds[pname])]
        sns.histplot(
            {pname: param_counters[pname], 'bounds' : pbounds[pname]},  # UGH add a fake histogram for the bounds, which sucks and means i have to kill the border for both of them, but otherwise it fucks up the bin widths
            bins=n_bins,
            edgecolor='darkred' if len(param_counters[pname]) < 5 else None,
            # binwidth=(xmax - xmin) / n_bins,  # this crashes sometimes in numpy, seems to be a bug
            palette={pname : '#78c2f1', 'bounds' : '#ffffff'},
        )
        plt.setp(ax.patches, linewidth=0)
        # for ptch in ax.patches:  # can set individual bin styles this way, but it seems harder to figure out which are the fake ones for the bounds
        #     print(ptch, ptch.get_x(), pbounds[pname])
        plt.legend([], [], frameon=False)  # remove legend since we only have one hist atm
        ax.set(xlabel=pname)
        # param_text = 'xscale %.1f\nxshift %.1f\nyscale %.1f' % (pfo["birth-response"].xscale, pfo["birth-response"].xshift, pfo["birth-response"].yscale)
        # fig.text(0.6, 0.25, param_text, fontsize=17)
        fn = "%s/chosen-%s-values.svg" % (plotdir, pname)
        plt.savefig(fn)
        return fn

    # ----------------------------------------------------------------------------------------
    print("    plotting chosen parameter values to %s" % plotdir)
    mpl_init()
    for sfn in glob.glob("%s/*.svg" % plotdir):
        os.remove(sfn)
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    if fnames is None:
        fnames = [[]]
    if len(fnames[-1]) > 0:  # add an empty row if there's already file names there
        fnames.append([])
    n_pval_list = [len(pvals) for pvals in param_counters.values()]
    if len(set(n_pval_list)) != 1:   # make sure nobody got incremented twice
        raise Exception('different number of counts for different parameters: %s' % ' '.join('%s %d'%(p, len(pvals)) for p, pvals in param_counters.items()))
    for pname in param_counters:
        fn = plot_param(pname)
        addfn(fnames, fn)

# ----------------------------------------------------------------------------------------
def get_resp_xvals(xbounds, nsteps):
    dx = (xbounds[1] - xbounds[0]) / nsteps
    xvals = list(np.arange(xbounds[0], 0, dx)) + list(np.arange(0, xbounds[1] + dx, dx))
    return xvals, dx  # NOTE len(xvals) is nsteps + 2

# ----------------------------------------------------------------------------------------
# if you already have xvals, you can pass them in directly and set xbounds and nsteps to None
def get_resp_vals(resp, xbounds, nsteps, normalize=False, xvals=None):
    if xvals is None:
        xvals, dx = get_resp_xvals(xbounds, nsteps)
    else:
        assert xbounds is None and nsteps is None
    rvals = [float(resp.λ_phenotype(x)) for x in xvals]
    if normalize:
        sumv = sum(rvals)
        rvals = [v / sumv for v in rvals]
    return rvals

# ----------------------------------------------------------------------------------------
def get_hist_vals(htmp, xbounds, xvals=None, normalize=False):
    if xvals is not None:
        assert xbounds is None
        rvals = [htmp.bin_contents[htmp.find_bin(x)] for x in xvals]
    elif xbounds is None:  # use bin centers
        rvals = [htmp.bin_contents[i] for i in htmp.ibiniter(include_overflows=False)]
    else:
        _, _, xvals, _ = get_resp_xlists(htmp, xbounds, None, is_hist=True)
        rvals = [htmp.bin_contents[htmp.find_bin(x)] for x in xvals]
    if normalize:
        sumv = sum(rvals)
        rvals = [v / sumv for v in rvals]
    return rvals

# ----------------------------------------------------------------------------------------
# add slope-type metrics (max slope and the x value at which it occurs, and the naive birth rate) corresponding to <tresp> to <tpfo>
def add_slope_vals(tresp, xbounds, tpfo, is_hist=False, nsteps=None, debug=False):
    if not is_hist and nsteps is None:
        nsteps = 120
    nsteps, xbounds, xvals, dxvlist = get_resp_xlists(tresp, xbounds, nsteps, is_hist=is_hist)
    rvals = get_hist_vals(tresp, None, xvals=xvals) if is_hist else get_resp_vals(tresp, None, None, xvals=xvals)
    assert len(rvals) == len(xvals) and len(xvals) == len(dxvlist)
    imaxdiff, rmaxdiff = sorted([(i, abs(rvals[i+1]-rvals[i])) for i in range(len(rvals)-1)], key=lambda p: p[1], reverse=True)[0]
    dxmax = (dxvlist[imaxdiff] + dxvlist[imaxdiff+1]) / 2  # the dx values are bin widths (if it's a hist), so we want the average bin width over the two bins we're subtracting
    x_init, init_birth = sorted([(x, v) for x, v in zip(xvals, rvals)], key=lambda p: abs(0-p[0]))[0]  # sort by nearness to 0
    tpfo['max_slope'], tpfo['x_max_slope'], tpfo['init_birth'] = rmaxdiff / dxmax, xvals[imaxdiff], init_birth
    tpfo['mean_val'], tpfo['max_val'], tpfo['max_diff'] = np.mean(rvals), max(rvals), max(rvals) - min(rvals)
    if debug:
        print('    slope difference values%s with %d x grid points:' % (' for hist' if is_hist else '', len(xvals)))
        def prlist(tn, lst, offset=False):
            print('        %5s %s%s' % (tn, '   ' if offset else '', ' '.join('%5.2f'%v for v in lst)))
        prlist('x', xvals)
        prlist('dx', dxvlist)
        prlist('resp', rvals)
        prlist('rdiff', [rvals[i+1]-rvals[i] for i in range(len(rvals)-1)], offset=True)
        print('      max slope %.2f   x max slope %.2f    init birth %.2f   mean val: %.2f   max val: %.2f   max diff: %.2f' % (tpfo['max_slope'], tpfo['x_max_slope'], tpfo['init_birth'], tpfo['mean_val'], tpfo['max_val'], tpfo['max_diff']))

# ----------------------------------------------------------------------------------------
# get all lists (x, y, dx, etc)
def get_resp_xlists(tresp, xbounds, nsteps, is_hist=False):
    if is_hist:
        assert nsteps is None
        bin_centers, bin_iter = tresp.get_bin_centers(ignore_overflows=True), tresp.ibiniter(include_overflows=False)
        if xbounds is None:
            xbounds = (tresp.xmin, tresp.xmax)
            nsteps = tresp.n_bins - 2
            xvals = bin_centers
            dxvlist = [tresp.binwidth(i) for i in bin_iter]
        else:
            xvals, dxvlist = zip(*[(c, tresp.binwidth(i)) for i, c in zip(bin_iter, bin_centers) if c > xbounds[0] and c < xbounds[1]])  # this gives you tuples rather than lists, which should be ok?
            nsteps = len(xvals) - 2
    else:
        xvals, dx = get_resp_xvals(xbounds, nsteps)
        dxvlist = [dx for _ in xvals]
    assert len(xvals) == len(dxvlist)
    return nsteps, xbounds, xvals, dxvlist

# ----------------------------------------------------------------------------------------
# can set resp2 to a hist if you set resp2_is_hist (rect_area: old-style full rectangular area for denominator)
# NOTE not symmetric: resp1 is used in the denominator (i.e. it should be the true response)
def resp_fcn_diff(resp1, resp2, xbounds, dont_normalize=False, nsteps=40, resp2_is_hist=False, rect_area=False, debug=False):  # see also nn.curve_loss()
    # ----------------------------------------------------------------------------------------
    def prdbg():
        def prvls(lbl, vlist, p=3):
            print('     %10s'%lbl, '  '.join(('%7.'+str(p)+'f')%v for v in vlist))
        if resp2_is_hist:
            print('   %d bins with xbounds %.3f to %.3f (widths %s):' % (resp2.n_bins, resp2.xmin, resp2.xmax, ' '.join('%.2f'%w for w in dxvlist)))
        else:
            print('   %d bins with xbounds %.3f to %.3f (width %.3f):' % (nsteps, xbounds[0], xbounds[1], dxvlist[0]))
        prvls('x', xvals, p=1)
        diffs = [v1-v2 for v1, v2 in zip(vals1, vals2)]
        vlists = [('v1', vals1), ('v2', vals2), ('diff', diffs), ('diff area', [abs(d)*dx for d, dx in zip(diffs, dxvlist)])]
        if not rect_area:
            vlists.append(('denom area', area_vlist))
        for l, vl in vlists:
            prvls(l, vl)
        if not dont_normalize:
            print('   normd diff: %.3f / %.3f = %.3f' % (sumv, area_val, sumv / area_val))
    # ----------------------------------------------------------------------------------------
    nsteps, xbounds, xvals, dxvlist = get_resp_xlists(resp2, xbounds, nsteps, is_hist=resp2_is_hist)  # resp2 is the one that can be a hist, so may as well use it for this
    if resp2_is_hist:
        vals1, vals2 = get_resp_vals(resp1, None, None, xvals=xvals), get_hist_vals(resp2, None, xvals=xvals)
    else:
        vals1, vals2 = [get_resp_vals(r, xbounds, nsteps) for r in [resp1, resp2]]
    assert len(vals1) == len(xvals) and len(vals2) == len(xvals)
    sumv = sum(abs(v1-v2) * dx for v1, v2, dx in zip(vals1, vals2, dxvlist))
    return_val = sumv
    if not dont_normalize:
        if rect_area:
            area_val = abs(xbounds[1] - xbounds[0]) * abs(max(vals1+vals2) - min(vals1+vals2))  # divide area between curves by area of rectangle defined by xbounds and min/max of either fcn
        else:
            # area_vlist = [max([abs(v1), abs(v2)]) * dx for v1, v2, dx in zip(vals1, vals2, dxvlist)]
            area_vlist = [abs(v1) * dx for v1, dx in zip(vals1, dxvlist)]
            area_val = sum(area_vlist)  # area between furthest curve and x axis (in this increment of dx) (don't really need dx, but maybe it makes it more intuitive)
        return_val /= area_val
    if debug:
        prdbg()
    return return_val

# ----------------------------------------------------------------------------------------
def get_rvals(pfo, xvals, xbounds, nsteps):
    if 'birth-hist' in pfo: # is_hist
        rvals = get_hist_vals(pfo['birth-hist'], None, xvals=xvals)
    else:
        rvals = get_resp_vals(pfo['birth-response'], xbounds, nsteps)
    return rvals

# ----------------------------------------------------------------------------------------
def resp_plot(bresp, ax, xbounds, alpha=0.8, color="#990012", nsteps=40, linewidth=3, linestyle='-'):
    xvals, dx = get_resp_xvals(xbounds, nsteps)
    rvals = get_resp_vals(bresp, xbounds, nsteps)
    data = {"affinity": xvals, "lambda": rvals}
    ax.set(xlabel=pltlabels.get('affinity')) #, ylabel='lambda')
    sns.lineplot(data, x="affinity", y="lambda", ax=ax, linewidth=linewidth, linestyle=linestyle, color=color, alpha=alpha)

# ----------------------------------------------------------------------------------------
# returns the element from <pfo_list> with minimal "distance" to the others
def get_median_curve(pfo_list, xbounds, is_hist=False, nsteps=50, debug=False):
    # ----------------------------------------------------------------------------------------
    def diff_fcn(r1vals, r2vals):
        assert len(r1vals) == len(r2vals)
        return sum((r2 - r1)**2 for r1, r2 in zip(r1vals, r2vals))
    # ----------------------------------------------------------------------------------------
    nsteps, xbounds, xvals, dxvlist = get_resp_xlists(pfo_list[0]['birth-hist'] if is_hist else None, xbounds, nsteps, is_hist=is_hist)
    if debug:
        print('   finding median among %d %s' % (len(pfo_list), 'hists' if is_hist else 'curves'))
        print('      x           %s' % ' '.join('%5.1f'%x for x in xvals))
    rvlists = []
    for pfo in pfo_list:
        rvals = get_rvals(pfo, xvals, xbounds, nsteps)
        rvlists.append(rvals)
    sumvals = [sum(diff_fcn(r1vals, r2vals) for r2vals in rvlists) for r1vals in rvlists]  # don't think there's any point in skipping the current one
    min_sum = min(sumvals)
    if debug:
        for icv, rvals in enumerate(rvlists):
            print('    %3d  %s   %s' % (icv, color('red' if min_sum==sumvals[icv] else None, '%6.1f'%sumvals[icv], width=5), ' '.join('%5.2f'%r for r in rvlists[icv])))
    imins = [i for i, s in enumerate(sumvals) if s==min_sum]
    if len(imins) > 1:
        print('    %s %d-degenerate median among %d curves' % (wrnstr(), len(imins), len(pfo_list)))
    return pfo_list[imins[0]]

# ----------------------------------------------------------------------------------------
def group_by_xvals(xvals, yvals, xbins, skip_overflows=False, debug=False):  # NOTE a lot of this is copied from partis hist.py
    htmp = Hist(xmin=xbins[0], xmax=xbins[-1], xbins=xbins, n_bins=len(xbins)-1)
    bin_xvals, bin_yvals = [[] for _ in htmp.low_edges], [[] for _ in htmp.low_edges]  # x one is just for debug
    for xv, yv in zip(xvals, yvals):
        ibin = htmp.find_bin(xv)
        bin_xvals[ibin].append(xv)
        bin_yvals[ibin].append(yv)
    if debug:
        print('             ibin  lo      N    mean   +/-      min    max')
    for ibin in htmp.ibiniter(not skip_overflows):
        ncont = len(bin_yvals[ibin])
        oustr = 'under' if ibin==0 else ('over' if ibin==htmp.n_bins+1 else '')
        if ncont == 0:
            if debug:
                print('      %5s  %2d  %6.2f  %3d' % (oustr, ibin, htmp.low_edges[ibin], ncont))
            continue
        # NOTE might be better to weight x position by x values
        htmp.set_ibin(ibin, np.mean(bin_yvals[ibin]), np.std(bin_yvals[ibin], ddof=1) / math.sqrt(len(bin_yvals[ibin])) if ncont>1 else 0)
        if debug:
            print('      %5s  %2d  %6.2f  %3d   %5.2f %5.2f     %5.2f  %5.2f' % (oustr, ibin, htmp.low_edges[ibin], ncont, htmp.bin_contents[ibin], htmp.errors[ibin], min(bin_xvals[ibin]), max(bin_xvals[ibin])))
    if debug:
        print(htmp)
    return htmp

# ----------------------------------------------------------------------------------------
def plot_curve_confidence_bands(pfo_list, xbounds, nsteps=50):
    curve_vals = []
    xvals, dx = get_resp_xvals(xbounds, nsteps)
    for ipf, pfo in enumerate(pfo_list):
        rvals = get_rvals(pfo, xvals, xbounds, nsteps)
        curve_vals.append(rvals)
    lower_1sigma, upper_1sigma = [np.percentile(curve_vals, p, axis=0) for p in [16, 84]]
    plt.fill_between(xvals, lower_1sigma, upper_1sigma, color='steelblue', alpha=0.3)
    lower_2sigma, upper_2sigma = [np.percentile(curve_vals, p, axis=0) for p in [2, 97.5]]
    plt.fill_between(xvals, lower_2sigma, upper_2sigma, color='skyblue', alpha=0.3)

# ----------------------------------------------------------------------------------------
def plot_many_curves(plotdir, plotname, pfo_list, titlestr=None, affy_vals=None, colors=None, ci_pfos=None,
                     add_sigmoid_true_pred_text=False, add_per_bin_true_pred_text=False, add_pred_text=False, param_text_pfos=None,
                     diff_vals=None, pred_xvals=None, pred_yvals=None, xbounds=None, ybounds=None, xbins=affy_bins, default_xbounds=None,
                     nonsense_affy_val=-99, median_pfo=None, pred_hists=None, alphas=None):
    if colors is not None and len(colors) != len(pfo_list):
        raise Exception('colors %d different length to pfo_list %d' % (len(colors), len(pfo_list)))
    if alphas is not None and len(alphas) != len(pfo_list):
        raise Exception('alphas %d different length to pfo_list %d' % (len(alphas), len(pfo_list)))
    if default_xbounds is None:
        default_xbounds = [-2.5, 3]
    mpl_init()
    fig, ax = plt.subplots()
    if pred_xvals is not None and pred_yvals is not None:  # only used for per-cell prediction
        if nonsense_affy_val in pred_xvals:
            print('    %s removing %d / %d nonsense affinity values' % (wrnstr(), pred_xvals.count(nonsense_affy_val), len(pred_xvals)))
            new_xvals, new_yvals = [], []
            for xv, yv in zip(pred_xvals, pred_yvals):
                if xv == nonsense_affy_val:
                    continue
                new_xvals.append(xv)
                new_yvals.append(yv)
            pred_xvals, pred_yvals = new_xvals, new_yvals
        # # binned with mean/std err (this is ok, but it kind of sucks to have binning, and we don't need it for individual tree plots)
        htmp = group_by_xvals(pred_xvals, pred_yvals, xbins) #, skip_overflows=True)
        htmp.mpl_plot(ax, square_bins=True, remove_empty_bins=True, color='darkred', no_vertical_bin_lines=True)
        ax.scatter(pred_xvals, pred_yvals, color='#2b65ec', alpha=0.45, marker='.', s=85)
        if xbounds is None:  # maybe should take more restrictive of the two?
            xbounds = [mfn(pred_xvals) for mfn in [min, max]]
    if pred_hists is not None:
        for pfo in pred_hists:
            pfo['birth-hist'].mpl_plot(ax, square_bins=True, remove_empty_bins=True, color='#1f77b4', no_vertical_bin_lines=True, alpha=0.6 if len(pred_hists)==1 else 0.1, linewidth=1.5 if len(pred_hists)>1 else 3, errors=False)
        ax.set(xlabel=pltlabels.get('affinity'), ylabel='lambda')
    if ci_pfos is not None:
        plot_curve_confidence_bands(ci_pfos, xbounds=default_xbounds)
    for ipf, pfo in enumerate(pfo_list):
        alpha = (0.1 if len(pfo_list)>5 else 0.5) if alphas is None else alphas[ipf]
        resp_plot(pfo['birth-response'], ax, alpha=alpha, color='#990012' if colors is None else colors[ipf], xbounds=pfo['xbounds'] if 'xbounds' in pfo and pfo['xbounds'] is not None else default_xbounds, linewidth=1 if colors is None else 2)  # it's important to use each curve's own xbounds to plot it, so that each curve only gets plotted over x values at which its gc had affinity values
    if median_pfo is not None:
        if 'birth-hist' in median_pfo:  # it's a hist
            median_pfo['birth-hist'].mpl_plot(ax, square_bins=True, remove_empty_bins=True, color='#ff7f0e', no_vertical_bin_lines=True, alpha=0.8, linewidth=3, errors=False)
        else:  # it's a response
            resp_plot(median_pfo['birth-response'], ax, alpha=0.8, color='#ff7f0e', xbounds=default_xbounds, linewidth=2, linestyle='--')  # it's important to use each curve's own xbounds to plot it, so that each curve only gets plotted over x values at which its gc had affinity values
        # from gcdyn.poisson import SigmoidResponse
        # tmp_params = [1.6, 2, 7]
        # resp_plot(SigmoidResponse(xscale=tmp_params[0], xshift=tmp_params[1], yscale=tmp_params[2], yshift=0), ax, alpha=0.3, color='blue', xbounds=default_xbounds, linewidth=1, linestyle='--')  # it's important to use each curve's own xbounds to plot it, so that each curve only gets plotted over x values at which its gc had affinity values
    ax2 = None
    if affy_vals is not None:
        if nonsense_affy_val in affy_vals:
            print('    %s removing %d / %d nonsense affinity values' % (wrnstr(), affy_vals.count(nonsense_affy_val), len(affy_vals)))
            affy_vals = [v for v in affy_vals if v != nonsense_affy_val]
        ax2 = ax.twinx()
        affy_plot({'all' : affy_vals}, ax2, color='#808080', alpha=0.3 if any(l is not None for l in [pfo_list, pred_hists]) else None) #, xbounds=default_xbounds)
        # if xbounds is None:  # maybe should take more restrictive of the two?
        #     xbounds = [mfn(affy_vals) for mfn in [min, max]]
    if param_text_pfos is not None:  # first one should be the true one
        if len(param_text_pfos) < 2:
            param_text_pfos.append(None)
        assert len(param_text_pfos) == 2
        param_text_pfos = [p if p!='median' else median_pfo for p in param_text_pfos]
        add_param_text(fig, param_text_pfos[0], inf_pfo=param_text_pfos[1], diff_vals=diff_vals, upper_left=True)
    if ybounds is not None and ybounds[0] != ybounds[1]:
        ax.set_ylim(ybounds[0], ybounds[1])
    if xbounds is not None and xbounds[0] != xbounds[1]:
        ax.set_xlim(xbounds[0], xbounds[1])
        if ax2 is not None:
            ax2.set_xlim(xbounds[0], xbounds[1])
    # ax.set_yscale('log')
    ax.set(title='%d responses' % len(pfo_list) if titlestr is None else titlestr)
    fn = "%s/%s.svg" % (plotdir, plotname)
    plt.savefig(fn)
    plt.close()
    return fn

# ----------------------------------------------------------------------------------------
# NOTE duplicates a lot of plot_all_diffs()
def affy_plot(affy_vals, ax, n_bins=50, xbounds=None, color=None, alpha=0.6):  # if xbounds is set, we *remove* affy values outside of those bounds (since it seems we can't control the sns binning/bounds)
    all_vals = [v for vlist in affy_vals.values() for v in vlist]
    if len(set(all_vals)) == 1:
        print('    %s all affinity values the same, can\'t plot (for some reason numpy histogram barfs)' % color('yellow', 'warning'))
        return
    if xbounds is not None:
        all_vals = [v for v in all_vals if v > xbounds[0] and v < xbounds[1]]
    xmin, xmax = [mfn(all_vals) for mfn in [min, max]]
    sns.histplot(affy_vals if len(affy_vals)>1 else all_vals, ax=ax, multiple="stack", binwidth=(xmax - xmin) / n_bins, color=color, alpha=alpha, linewidth=0) #, edgecolor='k')
    # print(sns.color_palette().as_hex())  # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# ----------------------------------------------------------------------------------------
# NOTE duplicates a lot of affy_plot
def plot_all_diffs(plotdir, plotname, curve_diffs, n_bins=30, xbounds=[0, 1], n_skipped_diffs=None):
    mpl_init()
    fig, ax = plt.subplots()
    all_vals = [v for vlist in curve_diffs.values() for v in vlist if v is not None]
    range_list = (xbounds+all_vals) if max(all_vals)-min(all_vals)>0.1 else all_vals  # if all_vals has a very narrow spread, then a bug in sns causes a crash (when range of all_vals is too small compared to bin width) [kinda guessing on 0.1, it may need to be adjusted]
    xmin, xmax = [mfn(range_list) for mfn in [min, max]]
    sns.histplot({s : vals for s, vals in curve_diffs.items() if len(vals)>0} if len(curve_diffs)>1 else all_vals, multiple='dodge', binwidth=(xmax - xmin) / n_bins)
    titlestr = ''
    for ism, smpl in enumerate([s for s, v in sorted(curve_diffs.items()) if len(v)>0]):
        fig.text(0.55, 0.6-0.065*ism, '%s:  mean %.3f'%(smpl, np.mean(curve_diffs[smpl])), fontsize=17)
        titlestr += '%s%s: %d%s%s' % ('' if len(titlestr)==0 else ' (', smpl, len(curve_diffs[smpl]), ' responses' if len(titlestr)==0 else '', '' if len(titlestr)==0 else ')')
        if n_skipped_diffs is not None and n_skipped_diffs[smpl] > 0:
            fig.text(0.475, 0.45-0.07*ism, '%s: skipped %d'%(smpl, n_skipped_diffs[smpl]), fontsize=20)
    ax.set(title=titlestr, xlabel='curve difference loss')
    ax.set_yscale('log')
    fn = "%s/%s.svg" % (plotdir, plotname)
    plt.savefig(fn)
    plt.close()
    return fn

# ----------------------------------------------------------------------------------------
def add_param_text(fig, true_pfo, inf_pfo=None, diff_vals=None, upper_left=False, titlestr=None):
    # ----------------------------------------------------------------------------------------
    def get_pval(tpfo, pname):
        if tpfo is None:
            return None
        elif pname in tpfo:
            return tpfo[pname]
        elif 'birth-response' in tpfo and hasattr(tpfo['birth-response'], pname):
            return getattr(tpfo['birth-response'], pname)
        else:
            return None
    # ----------------------------------------------------------------------------------------
    def gpn(pname):
        return pname.replace('max_', '').replace('init_', '').replace('_', ' ')
    # ----------------------------------------------------------------------------------------
    def ptext(pname):
        true_pval, inf_pval = [get_pval(tpfo, pname) for tpfo in [true_pfo, inf_pfo]]
        rstr = ''
        if true_pval is not None:
            rstr += '%s %.1f' % (gpn(pname), true_pval)
        if inf_pval is not None:
            rstr += '%s (%.1f)' % (gpn(pname) if rstr=='' else '', inf_pval)
        return rstr
    # ----------------------------------------------------------------------------------------
    xv, yv = (0.6, 0.25) if inf_pfo is None and not upper_left else (0.19, 0.7 if diff_vals is None else 0.65)
    plist = ['xscale', 'xshift', 'yscale', 'yshift', 'x_ceil_start'] if 'birth-response' in non_none([true_pfo, inf_pfo]) else ['max_slope', 'x_max_slope', 'init_birth']
    param_text = [ptext(p) for p in plist]
    param_text = [t for t in param_text if t != '']
    if len(param_text) > 3:
        yv -= 0.05 * (len(param_text) - 3)
    if diff_vals is not None:
        for dv in diff_vals:
            if hasattr(dv, 'keys'):
                ptxt = 'diff %.2f' % dv['diff']
                if 'xbounds' in dv:
                    ptxt += ' (%.1f %.1f)' % (dv['xbounds'][0], dv['xbounds'][1])
                param_text.append(ptxt)
            else:
                param_text.append('diff %.2f' % dv)
    if titlestr is not None:
        param_text.insert(0, titlestr)
        yv -= 0.05
    fig.text(xv, yv, '\n'.join(param_text), fontsize=17)

# ----------------------------------------------------------------------------------------
def plot_phenotype_response(plotdir, pfo_list, n_to_plot=20, bundle_size=1, fnames=None):
    # ----------------------------------------------------------------------------------------
    def get_afplot(pfo, ax, itree):
        leaves = list(pfo["tree"].iter_leaves())
        leaf_vals = [n.x for n in leaves]
        int_vals = [n.x for n in pfo["tree"].iter_descendants() if n not in leaves]
        affy_plot({'internal' : int_vals, 'leaf' : leaf_vals}, ax)
        ax.set(title="itree %d (%d nodes)"%(itree, len(leaf_vals+int_vals)))
        return [mfn(leaf_vals+int_vals) for mfn in [min, max]]
    # ----------------------------------------------------------------------------------------
    def plt_single_tree(itree, pfo):
        plt.clf()
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        xbounds = get_afplot(pfo, ax2, itree)
        resp_plot(pfo['birth-response'], ax, xbounds)
        add_param_text(fig, pfo)
        fn = "%s/trees-%d.svg" % (plotdir, itree)
        plt.savefig(fn)
        return fn
    # ----------------------------------------------------------------------------------------
    print("    plotting trees to %s" % plotdir)
    for sfn in glob.glob("%s/*.svg" % plotdir):
        os.remove(sfn)
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    if fnames is None:
        fnames = [[]]
    # if len(fnames[-1]) > n_columns:  # add an empty row if there's already file names there
    #     fnames.append([])

    n_to_plot = min(len(pfo_list), n_to_plot)
    if bundle_size == 1:
        plt_indices = range(n_to_plot)
    else:
        plt_indices = range(0, min(n_to_plot * bundle_size, len(pfo_list)), bundle_size)

    fn = plot_many_curves(plotdir, 'all-responses', [pfo_list[i] for i in plt_indices])
    addfn(fnames, fn)
    for itree in plt_indices:
        fn = plt_single_tree(itree, pfo_list[itree])
        addfn(fnames, fn)

# ----------------------------------------------------------------------------------------
def memory_usage_fraction(
    extra_str="", debug=False
):  # return fraction of total system memory that this process is using (as always with memory things, this is an approximation)
    if platform.system() != "Linux":
        print(
            "\n  note: utils.memory_usage_fraction() needs testing on platform '%s' to make sure unit conversions don't need changing"
            % platform.system()
        )
    current_usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # kb
    total = (
        float(psutil.virtual_memory().total) / 1000.0
    )  # returns bytes, then convert to kb
    if debug:
        print(
            "  %susing %.0f / %.0f MB = %.3f%%"
            % (
                extra_str,
                current_usage / 1000,
                total / 1000,
                100 * current_usage / total,
            )
        )
    return current_usage / total


# ----------------------------------------------------------------------------------------
def limit_procs(procs, n_max_procs, sleep_time=1, debug=False):
    """Count number of <procs> that are currently running, and sleep until it's less than <n_max_procs>."""

    def n_running_jobs():
        return [p.poll() for p in procs].count(None)

    n_jobs = n_running_jobs()
    while n_jobs >= n_max_procs:
        if debug:
            print("%d (>=%d) running jobs" % (n_jobs, n_max_procs))
        time.sleep(sleep_time)
        n_jobs = n_running_jobs()


# ----------------------------------------------------------------------------------------
def make_html(
    plotdir,
    n_columns=3,
    extension="svg",
    fnames=None,
    title="foop",
    bgcolor="000000",
    new_table_each_row=False,
    htmlfname=None,
    extra_links=None,
):
    """make an html file displaying all the svg (by default) files in <plotdir>"""
    if fnames is not None:  # make sure it's formatted properly
        for rowfnames in fnames:
            if not isinstance(rowfnames, list):
                raise Exception(
                    "each entry in fnames should be a list of strings, but got a %s: %s"
                    % (type(rowfnames), rowfnames)
                )
            for fn in rowfnames:
                if not isinstance(fn, str):  # , unicode)
                    raise Exception(
                        "each entry in each row should be a string (file name), but got a %s: %s"
                        % (type(fn), fn)
                    )
    if plotdir[-1] == "/":  # remove trailings slash, if present
        plotdir = plotdir[:-1]
    if not os.path.exists(plotdir):
        raise Exception("plotdir %s d.n.e." % plotdir)
    dirname = os.path.basename(plotdir)
    extra_link_str = ""
    if extra_links is not None:
        extra_link_str = " ".join(
            ["<a href=%s>%s</a>" % (url, name) for name, url in extra_links]
        )
    lines = [
        '<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2//EN>',
        "<html>",
        "<head><title>" + title + "</title></head>",
        '<body bgcolor="%s">' % bgcolor,
        '<h3 style="text-align:left; color:DD6600;">' + title + "</h3>",
        extra_link_str,
        "<table>",
        "<tr>",
    ]

    def add_newline(lines, header=None):
        if new_table_each_row:
            endlines, startlines = ["</tr>", "</table>"], ["<table>", "<tr>"]
        else:
            endlines, startlines = ["</tr>"], ["<tr>"]
        lines += endlines
        if header is not None:
            lines += ['<h3 style="text-align:left; color:DD6600;">' + header + "</h3>"]
        lines += startlines

    def add_fname(
        lines, fullfname
    ):  # NOTE <fullname> may, or may not, be a base name (i.e. it might have a subdir tacked on the left side)
        fname = fullfname.replace(plotdir, "").lstrip("/")
        if (
            htmlfname is None
        ):  # dirname screws it up if we're specifying htmlfname explicitly, since then the files are in a variety of different subdirs
            fname = dirname + "/" + fname
        line = (
            '<td><a target="_blank" href="'
            + fname
            + '"><img src="'
            + fname
            + '" alt="'
            + fname
            + '" width="100%"></a></td>'
        )
        lines.append(line)

    # if <fnames> wasn't used to tell us how to group them into rows, try to guess based on the file base names
    if fnames is None:
        fnamelist = [
            os.path.basename(fn)
            for fn in sorted(glob.glob(plotdir + "/*." + extension))
        ]
        fnames = []

        # then do the rest in groups of <n_columns>
        while len(fnamelist) > 0:
            fnames.append(fnamelist[:n_columns])
            fnamelist = fnamelist[n_columns:]

    # write the meat of the html
    for rowlist in fnames:
        if "header" in rowlist:
            if len(rowlist) != 2:
                raise Exception(
                    "malformed header row list in fnames (should be len 2 but got %d): %s"
                    % (len(rowlist), rowlist)
                )
            add_newline(lines, header=rowlist[1])
            continue
        for fn in rowlist:
            add_fname(lines, fn)
        add_newline(lines)

    lines += ["</tr>", "</table>", "</body>", "</html>"]

    if htmlfname is None:
        htmlfname = (
            os.path.dirname(plotdir) + "/" + dirname + ".html"
        )  # more verbose than necessary
    with open(htmlfname, "w") as htmlfile:
        htmlfile.write("\n".join(lines))
    # subprocess.check_call(['chmod', '664', htmlfname])

# ----------------------------------------------------------------------------------------
def pad_lines(lstr, extra_str='            '):
    return '\n'.join(extra_str+l for l in lstr.split('\n'))

# ----------------------------------------------------------------------------------------
def print_as_dtree(etree, width=250, label_fcn=None, extra_str='            '):
    # # ete tree version (better than __str__()), although it still can't show distance:
    # print(pad_lines(etree.get_ascii(show_internal=True), extra_str=extra_str))
    import dendropy
    dtree = dendropy.Tree.get_from_string(etree.write(format=1), 'newick', suppress_internal_node_taxa=False, preserve_underscores=True)
    print(get_ascii_tree(dtree, extra_str=extra_str, label_fcn=label_fcn, width=width))

# ----------------------------------------------------------------------------------------
def get_ascii_tree(dendro_tree, extra_str='', width=200, schema='newick', label_fcn=None):
    """
        AsciiTreePlot docs (don't show up in as_ascii_plot()):
            plot_metric : str
                A string which specifies how branches should be scaled, one of:
                'age' (distance from tips), 'depth' (distance from root),
                'level' (number of branches from root) or 'length' (edge
                length/weights).
            show_internal_node_labels : bool
                Whether or not to write out internal node labels.
            leaf_spacing_factor : int
                Positive integer: number of rows between each leaf.
            width : int
                Force a particular display width, in terms of number of columns.
            node_label_compose_fn : function object
                A function that takes a Node object as an argument and returns
                the string to be used to display it.
    """
    if all(l.distance_from_root()==0 for l in dendro_tree.leaf_node_iter()):  # we really want the max height, but since we only care whether it's zero or not this is the same
        return '%szero height' % extra_str
    # elif: get_n_nodes(dendro_tree) > 1:  # not sure if I really need this if any more (it used to be for one-leaf trees (and then for one-node trees), but the following code (that used to be indented) seems to be working fine on one-leaf, one-node, and lots-of-node trees a.t.m.)

    start_char, end_char = '', ''
    def compose_fcn(x):
        if x.taxon is not None:  # if there's a taxon defined, use its label
            lb = x.taxon.label
        elif x.label is not None:  # use node label
            lb = x.label
        else:
            lb = 'o'
        if label_fcn is not None:
            lb = label_fcn(lb)
        return '%s%s%s' % (start_char, lb, end_char)
    dendro_str = dendro_tree.as_ascii_plot(width=width, plot_metric='length', show_internal_node_labels=True, node_label_compose_fn=compose_fcn)
    special_chars = [c for c in reversed(string.punctuation) if c not in set(dendro_str)]  # find some special characters that we can use to identify the start and end of each label (could also use non-printable special characters, but it shouldn't be necessary)
    if len(special_chars) >= 2:  # can't color them directly, since dendropy counts the color characters as printable
        start_char, end_char = special_chars[:2]  # NOTE the colors get screwed up when dendropy overlaps labels (or sometimes just straight up strips stuff), which it does when it runs out of space
        dendro_str = dendro_tree.as_ascii_plot(width=width, plot_metric='length', show_internal_node_labels=True, node_label_compose_fn=compose_fcn)  # call again after modiying compose fcn (kind of wasteful to call it twice, but it shouldn't make a difference)
        dendro_str = dendro_str.replace(start_char, Colors['blue']).replace(end_char, Colors['end'] + '  ')
    else:
        print('  %s can\'t color tree, no available special characters in get_ascii_tree()' % color('red', 'note:'))
    if len(list(dendro_tree.preorder_node_iter())) == 1:
        extra_str += ' (one node)'
    return_lines = [('%s%s' % (extra_str, line)) for line in dendro_str.split('\n')]
    return '\n'.join(return_lines)

# ----------------------------------------------------------------------------------------
def get_etree(fname=None, treestr=None):  # specify either fname or treestr
    if treestr is None:
        assert fname is not None
        with open(fname) as tfile:
            treestr = tfile.read() #.replace('[&R]', '').strip()
    if len(treestr.split()) == 2 and treestr.split()[0] in ['[&U]', '[&R]']:  # dumbest #$!#$#ing format in the goddamn world (ete barfs on other programs' rooting information)
        treestr = treestr.split()[1]
    return ete3.Tree(treestr, format=1, quoted_node_names=True)
    # return bdms.TreeNode(newick=treestr, format=1, quoted_node_names=True)

# ----------------------------------------------------------------------------------------
def hamming_distance(seq1, seq2, amino_acid=False):
    assert len(seq1) == len(seq2)
    ambig_bases = 'X' if amino_acid else 'N-'
    return sum(x != y for x, y in zip(seq1.upper(), seq2.upper()) if x not in ambig_bases and y not in ambig_bases)

# ----------------------------------------------------------------------------------------
def write_fasta(ofn, seqfos):
    with open(ofn, "w") as asfile:
        for sfo in seqfos:
            asfile.write(">%s\n%s\n" % (sfo["name"], sfo["seq"]))

# ----------------------------------------------------------------------------------------
def getsuffix(fname):  # suffix, including the dot
    if len(os.path.splitext(fname)) != 2:
        raise Exception('couldn\'t split %s into two pieces using dot' % fname)
    return os.path.splitext(fname)[1]

# ----------------------------------------------------------------------------------------
# if <look_for_tuples> is set, look for uids that are actually string-converted python tuples, and add each entry in the tuple as a duplicate sequence. Can also pass in a list <tuple_info> if you need to do more with the info afterwards (this is to handle gctree writing fasta files with broken names; see usage also in datascripts/meta/taraki-XXX)
def read_fastx(fname, name_key='name', seq_key='seq', add_info=True, dont_split_infostrs=False, sanitize_uids=False, sanitize_seqs=False, queries=None, n_max_queries=-1, istartstop=None, ftype=None, n_random_queries=None, look_for_tuples=False, tuple_info=None):
    if ftype is None:
        suffix = getsuffix(fname)
        if suffix == '.fa' or suffix == '.fasta':
            ftype = 'fa'
        elif suffix == '.fq' or suffix == '.fastq':
            ftype = 'fq'
        else:
            raise Exception('unhandled file type: %s' % suffix)

    finfo = []
    iline = -1  # index of the query/seq that we're currently reading in the fasta
    n_fasta_queries = 0  # number of queries so far added to <finfo> (I guess I could just use len(finfo) at this point)
    missing_queries = set(queries) if queries is not None else None
    already_printed_forbidden_character_warning, already_printed_warn_char_warning = False, False
    with open(fname) as fastafile:
        startpos = None
        while True:
            if startpos is not None:  # rewind since the last time through we had to look to see when the next header line appeared
                fastafile.seek(startpos)
            headline = fastafile.readline()
            if not headline:
                break
            if headline.strip() == '':  # skip a blank line
                headline = fastafile.readline()

            if ftype == 'fa':
                if headline[0] != '>':
                    raise Exception('invalid fasta header line in %s:\n    %s' % (fname, headline))
                headline = headline.lstrip('>')

                seqlines = []
                nextline = fastafile.readline()
                while True:
                    if not nextline:
                        break
                    if nextline[0] == '>':
                        break
                    else:
                        startpos = fastafile.tell()  # i.e. very line that doesn't begin with '>' increments <startpos>
                    seqlines.append(nextline)
                    nextline = fastafile.readline()
                seqline = ''.join([l.strip() for l in seqlines]) if len(seqlines) > 0 else None
            elif ftype == 'fq':
                if headline[0] != '@':
                    raise Exception('invalid fastq header line in %s:\n    %s' % (fname, headline))
                headline = headline.lstrip('@')

                seqline = fastafile.readline()  # NOTE .fq with multi-line entries isn't supported, since delimiter characters are allowed to occur within the quality string
                plusline = fastafile.readline().strip()
                if plusline[0] != '+':
                    raise Exception('invalid fastq quality header in %s:\n    %s' % (fname, plusline))
                qualityline = fastafile.readline()
            else:
                raise Exception('unhandled ftype %s' % ftype)

            if not seqline:
                break

            iline += 1
            if istartstop is not None:
                if iline < istartstop[0]:
                    continue
                elif iline >= istartstop[1]:
                    continue

            if dont_split_infostrs:  # if this is set, we let the calling fcn handle all the infostr parsing (e.g. for imgt germline fasta files)
                infostrs = headline
                uid = infostrs
            else:  # but by default, we split by everything that could be a separator, which isn't really ideal, but we're reading way too many different kinds of fasta files at this point to change the default
                # NOTE commenting this since it fucking breaks on the imgt fastas, which use a different fucking format and i don't even remember whose stupid format this was for WTF PEOPLE
                # if ';' in headline and '=' in headline:  # HOLY SHIT PEOPLE DON"T PUT YOUR META INFO IN YOUR FASTA FILES
                #     infostrs = [s1.split('=') for s1 in headline.strip().split(';')]
                #     uid = infostrs[0][0]
                #     infostrs = dict(s for s in infostrs if len(s) == 2)
                # else:
                infostrs = [s3.strip() for s1 in headline.split(' ') for s2 in s1.split('\t') for s3 in s2.split('|')]  # NOTE the uid is left untranslated in here
                uid = infostrs[0]
            if sanitize_uids and any(fc in uid for fc in forbidden_characters):
                if not already_printed_forbidden_character_warning:
                    print('  %s: found a forbidden character (one of %s) in sequence id \'%s\'. This means we\'ll be replacing each of these forbidden characters with a single letter from their name (in this case %s). If this will cause problems you should replace the characters with something else beforehand.' % (color('yellow', 'warning'), ' '.join(["'" + fc + "'" for fc in forbidden_characters]), uid, uid.translate(forbidden_character_translations)))
                    already_printed_forbidden_character_warning = True
                uid = uid.translate(forbidden_character_translations)
            if sanitize_uids and any(wc in uid for wc in warn_chars):
                if not already_printed_warn_char_warning:
                    print('  %s: found a character that may cause problems if doing phylogenetic inference (one of %s) in sequence id \'%s\' (only printing this warning on first occurence).' % (color('yellow', 'warning'), ' '.join(["'" + fc + "'" for fc in warn_chars]), uid))
                    already_printed_warn_char_warning = True

            if queries is not None:
                if uid not in queries:
                    continue
                missing_queries.remove(uid)

            seqfo = {name_key : uid, seq_key : seqline.strip().upper()}
            if add_info:
                seqfo['infostrs'] = infostrs
            if sanitize_seqs:
                seqfo[seq_key] = seqfo[seq_key].translate(ambig_translations).upper()
                if any(c not in alphabet for c in seqfo[seq_key]):
                    unexpected_chars = set([ch for ch in seqfo[seq_key] if ch not in alphabet])
                    raise Exception('unexpected character%s %s (not among %s) in input sequence with id %s:\n  %s' % (plural(len(unexpected_chars)), ', '.join([('\'%s\'' % ch) for ch in unexpected_chars]), alphabet, seqfo[name_key], seqfo[seq_key]))
            finfo.append(seqfo)

            n_fasta_queries += 1
            if n_max_queries > 0 and n_fasta_queries >= n_max_queries:
                break
            if queries is not None and len(missing_queries) == 0:
                break
    if n_max_queries > 0:
        print('    stopped after reading %d sequences from %s' % (n_max_queries, fname))
    if queries is not None:
        print('    only looked for %d specified sequences in %s' % (len(queries), fname))

    if n_random_queries is not None:
        if n_random_queries > len(finfo):
            print('  %s asked for n_random_queries %d from file with only %d entries, so just taking all of them (%s)' % (color('yellow', 'warning'), n_random_queries, len(finfo), fname))
            n_random_queries = len(finfo)
        finfo = np.random.choice(finfo, n_random_queries, replace=False)

    if look_for_tuples:  # this is because gctree writes broken fasta files with multiple uids in a line
        new_sfos, n_found = [], 0
        for sfo in finfo:
            headline = ' '.join(sfo['infostrs'])
            if any(c not in headline for c in '()'):
                new_sfos.append(sfo)
                continue
            istart, istop = [headline.find(c) for c in '()']
            try:
                nids = eval(headline[istart : istop + 1])
                if tuple_info is not None:
                    tuple_info.append(nids)
                n_found += 1
                print('         look_for_tuples: found %d uids in headline %s: %s' % (len(nids), headline, ' '.join(nids)))
                for uid in nids:
                    nsfo = copy.deepcopy(sfo)
                    nsfo['name'] = uid
                    new_sfos.append(nsfo)
            except:
                print('    %s failed parsing tuple from fasta line \'%s\' from %s' % (wrnstr(), headline, fname))
                new_sfos.append(sfo)
        if n_found > 0:
            print('      found %d seqfos with tuple headers (added %d net total seqs) in %s' % (n_found, len(new_sfos) - len(finfo), fname))
        finfo = new_sfos

    return finfo

# ----------------------------------------------------------------------------------------
def get_single_entry(tl, errmsg=None):  # adding this very late, so there's a lot of places it could be used
    if len(tl) != 1:
        raise Exception('length must be 1 in get_single_entry(), but got %d (%s)%s' % (len(tl), tl, '' if errmsg is None else ': %s'%errmsg))
    return tl[0]
def wrnstr():  # adding this very late, so could use it in a *lot* of places
    return color('yellow', 'warning')
eps = 1.0e-10  # if things that should be 1.0 are this close to 1.0, blithely keep on keepin on. kinda arbitrary, but works for the moment
def is_normed(probs, this_eps=eps, total=1.):
    if hasattr(probs, 'keys'):  # if it's a dict, call yourself with a list of the dict's values
        return is_normed([val for val in probs.values()], this_eps=this_eps)
    elif hasattr(probs, '__iter__'):  # if it's a list call yourself with their sum
        return is_normed(sum(probs), this_eps=this_eps)
    else:  # and if it's a float actually do what you're supposed to do
        return math.fabs(probs - total) < this_eps
def csv_wmode(mode='w'):
    if sys.version_info.major < 3:  # 2.7 csv module doesn't support unicode, this is the hackey fix
        return mode + 'b'
    else:
        return mode

# ----------------------------------------------------------------------------------------
# copied from partis
class Hist(object):
    """ a simple histogram """
    def __init__(self, n_bins=None, xmin=None, xmax=None, sumw2=False, xbins=None, template_hist=None, fname=None, value_list=None, weight_list=None, init_int_bins=False, xtitle='', ytitle='counts', title=''):
        # <xbins>: low edge of all non-under/overflow bins, plus low edge of overflow bin (i.e. low edge of every bin except underflow). Weird, but it's root conventions and it's fine
        self.low_edges, self.bin_contents, self.bin_labels = [], [], []
        self.xtitle, self.ytitle, self.title = xtitle, ytitle, title

        if fname is None:
            if init_int_bins:  # use value_list to initialize integer bins NOTE adding this very late, and i tink there's lots of places where it could be used
                assert value_list is not None
                xmin, xmax = min(value_list) - 0.5, max(value_list) + 0.5
                n_bins = xmax - xmin
            if template_hist is not None:
                assert n_bins is None and xmin is None and xmax is None and xbins is None  # specify *only* the template hist
                n_bins = template_hist.n_bins
                xmin, xmax = template_hist.xmin, template_hist.xmax
                xbins = template_hist.low_edges[1:]
            assert n_bins is not None
            assert xmin is not None and xmax is not None
            self.scratch_init(n_bins, xmin, xmax, sumw2=sumw2, xbins=xbins)
            if template_hist is not None and template_hist.bin_labels.count('') != len(template_hist.bin_labels):
                self.bin_labels = [l for l in template_hist.bin_labels]
        else:
            self.file_init(fname)

        if value_list is not None:
            if any(math.isnan(v) for v in value_list):
                raise Exception('nan value in value_list: %s' % value_list)
            if any(v < xmin or v >= xmax for v in value_list):  # probably because you forgot that xmax is low edge of overflow bin, so it's included in that
                # NOTE it would be nice to integrate this with hutils.make_hist_from_list_of_values() and hutils.make_hist_from_dict_of_counts()
                print('  %s value[s] %s outside bounds [%s, %s] in hist list fill' % (color('yellow', 'warning'), [v for v in value_list if v < xmin or v >= xmax], xmin, xmax))
            self.list_fill(value_list, weight_list=weight_list)

    # ----------------------------------------------------------------------------------------
    def scratch_init(self, n_bins, xmin, xmax, sumw2=False, xbins=None):
        self.n_bins = int(n_bins)
        self.xmin, self.xmax = float(xmin), float(xmax)
        self.errors = None if sumw2 else []
        self.sum_weights_squared = [] if sumw2 else None

        if xbins is not None:  # check validity of handmade bins
            if len(xbins) != self.n_bins + 1:
                raise Exception('misspecified xbins: should be n_bins + 1 (%d, i.e. the low edges of each non-under/overflow bin plus the low edge of the overflow bin) but got %d' % (self.n_bins + 1, len(xbins)))
            assert self.xmin == xbins[0]
            assert self.xmax == xbins[-1]
            if len(set(xbins)) != len(xbins):
                raise Exception('xbins has duplicate entries: %s' % xbins)

        dx = 0.0 if self.n_bins == 0 else (self.xmax - self.xmin) / float(self.n_bins)
        for ib in range(self.n_bins + 2):  # using root conventions: zero is underflow and last bin is overflow
            self.bin_labels.append('')
            if xbins is None:  # uniform binning
                self.low_edges.append(self.xmin + (ib-1)*dx)  # subtract one from ib so underflow bin has upper edge xmin. NOTE this also means that <low_edges[-1]> is the lower edge of the overflow
            else:  # handmade bins
                if ib == 0:
                    self.low_edges.append(xbins[0] - dx)  # low edge of underflow needs to be less than xmin, but is otherwise arbitrary, so just choose something that kinda makes sense
                else:
                    self.low_edges.append(xbins[ib-1])
            self.bin_contents.append(0.0)
            if sumw2:
                self.sum_weights_squared.append(0.)
            else:
                self.errors.append(0.)  # don't set the error values until we <write> (that is unless you explicitly set them with <set_ibin()>

    # ----------------------------------------------------------------------------------------
    def file_init(self, fname):
        self.errors, self.sum_weights_squared = [], []  # kill the unused one after reading file
        with open(fname, 'r') as infile:
            reader = csv.DictReader(infile)
            for line in reader:
                self.low_edges.append(float(line['bin_low_edge']))
                self.bin_contents.append(float(line['contents']))
                if 'sum-weights-squared' in line:
                    self.sum_weights_squared.append(float(line['sum-weights-squared']))
                if 'error' in line or 'binerror' in line:  # in theory I should go find all the code that writes these files and make 'em use the same header for this
                    assert 'sum-weights-squared' not in line
                    tmp_error = float(line['error']) if 'error' in line else float(line['binerror'])
                    self.errors.append(tmp_error)
                if 'binlabel' in line:
                    self.bin_labels.append(line['binlabel'])
                else:
                    self.bin_labels.append('')
                if 'xtitle' in line:  # should be the same for every line in the file... but this avoids complicating the file format
                    self.xtitle = line['xtitle']

        self.n_bins = len(self.low_edges) - 2  # file should have a line for the under- and overflow bins
        self.xmin, self.xmax = self.low_edges[1], self.low_edges[-1]  # *upper* bound of underflow, *lower* bound of overflow

        assert sorted(self.low_edges) == self.low_edges
        assert len(self.bin_contents) == len(self.low_edges)
        assert len(self.low_edges) == len(self.bin_labels)
        if len(self.errors) == 0:  # (re)set to None if the file didn't have errors listed
            self.errors = None
            assert len(self.sum_weights_squared) == len(self.low_edges)
        if len(self.sum_weights_squared) == 0:
            self.sum_weights_squared = None
            assert len(self.errors) == len(self.low_edges)

    # ----------------------------------------------------------------------------------------
    def getdict(self):  # get a dict suitable for writing to json/yaml file (ick! but i don't always want the hists to be in their own file) NOTE code reversing this is in test/cf-tree-metrics.py
        return {'n_bins' : self.n_bins, 'xmin' : self.xmin, 'xmax' : self.xmax, 'bin_contents' : self.bin_contents}

    # ----------------------------------------------------------------------------------------
    def is_overflow(self, ibin):  # return true if <ibin> is either the under or over flow bin
        return ibin in [0, self.n_bins + 1]

    # ----------------------------------------------------------------------------------------
    def overflow_contents(self):
        return self.bin_contents[0] + self.bin_contents[-1]

    # ----------------------------------------------------------------------------------------
    def ibiniter(self, include_overflows, reverse=False):  # return iterator over ibins (adding this late, so could probably be used in a lot of places that it isn't)
        if include_overflows:
            istart, istop = 0, self.n_bins + 2
        else:
            istart, istop = 1, self.n_bins + 1
        step = 1
        if reverse:
            itmp = istart
            istart = istop - 1
            istop = itmp - 1
            step = -1
        return list(range(istart, istop, step))

    # ----------------------------------------------------------------------------------------
    def set_ibin(self, ibin, value, error, label=None):
        """ set <ibin>th bin to <value> """
        self.bin_contents[ibin] = value
        if error is not None:
            if self.errors is None:
                raise Exception('attempted to set ibin error with none type <self.errors>')
            else:
                self.errors[ibin] = error
        if label is not None:
            self.bin_labels[ibin] = label

    # ----------------------------------------------------------------------------------------
    def fill_ibin(self, ibin, weight=1.0):
        """ fill <ibin>th bin with <weight> """
        self.bin_contents[ibin] += weight
        if self.sum_weights_squared is not None:
            self.sum_weights_squared[ibin] += weight*weight
        if self.errors is not None:
            if weight != 1.0:
                print('WARNING using errors instead of sumw2 with weight != 1.0 in Hist::fill_ibin()')
            self.errors[ibin] = math.sqrt(self.bin_contents[ibin])

    # ----------------------------------------------------------------------------------------
    def find_bin(self, value, label=None):  # if <label> is set, find ibin corresponding to bin label <label>
        """ find <ibin> corresponding to <value>. NOTE boundary is owned by the upper bin. """
        if label is not None:
            if label not in self.bin_labels:
                raise Exception('asked for label \'%s\' that isn\'t among bin labels: %s' % (label, self.bin_labels))
            return get_single_entry([i for i, l in enumerate(self.bin_labels) if l==label])
        if value < self.low_edges[0]:  # is it below the low edge of the underflow?
            return 0
        elif value >= self.low_edges[self.n_bins + 1]:  # or above the low edge of the overflow?
            return self.n_bins + 1
        else:
            for ib in range(self.n_bins + 2):  # loop over all the bins (including under/overflow)
                if value >= self.low_edges[ib] and value < self.low_edges[ib+1]:  # NOTE <ib> never gets to <n_bins> + 1 because we already get all the overflows above (which is good 'cause this'd fail with an IndexError)
                    return ib
        print(self)
        raise Exception('couldn\'t find bin for value %f (see lines above)' % value)

    # ----------------------------------------------------------------------------------------
    def fill(self, value, weight=1.0):
        """ fill bin corresponding to <value> with <weight> """
        self.fill_ibin(self.find_bin(value), weight)

    # ----------------------------------------------------------------------------------------
    def list_fill(self, value_list, weight_list=None):
        if weight_list is None:
            for value in value_list:
                self.fill(value)
        else:
            for value, weight in zip(value_list, weight_list):
                self.fill(value, weight=weight)

    # ----------------------------------------------------------------------------------------
    def get_extremum(self, mtype, xbounds=None, exclude_empty=False):  # NOTE includes under/overflows by default for max, but *not* for min
        if xbounds is None:
            if mtype == 'min':
                ibin_start, ibin_end = 1, self.n_bins + 1
            if mtype == 'max':
                ibin_start, ibin_end = 0, self.n_bins + 2
        else:
            ibin_start, ibin_end = [self.find_bin(b) for b in xbounds]

        if ibin_start == ibin_end:
            return self.bin_contents[ibin_start]

        ymin, ymax = None, None
        for ibin in range(ibin_start, ibin_end):
            if exclude_empty and self.bin_contents[ibin] == 0:
                continue
            if ymin is None or self.bin_contents[ibin] < ymin:
                ymin = self.bin_contents[ibin]
        for ibin in range(ibin_start, ibin_end):
            if ymax is None or self.bin_contents[ibin] > ymax:
                ymax = self.bin_contents[ibin]
        if ymin is None and exclude_empty:
            print('  %s couldn\'t find ymin for hist, maybe because <exclude_empty> was set (setting ymin arbitrarily to -99999)' % wrnstr())
            ymin = -99999
        assert ymin is not None and ymax is not None

        if mtype == 'min':
            return ymin
        elif mtype == 'max':
            return ymax
        else:
             assert False

    # ----------------------------------------------------------------------------------------
    def get_maximum(self, xbounds=None):  # NOTE includes under/overflows by default
        return self.get_extremum('max', xbounds=xbounds)

    # ----------------------------------------------------------------------------------------
    def get_minimum(self, xbounds=None, exclude_empty=False):  # NOTE does *not* include under/overflows by default (unlike previous fcn, since we expect under/overflows to be zero)
        return self.get_extremum('min', xbounds=xbounds, exclude_empty=exclude_empty)

    # ----------------------------------------------------------------------------------------
    def get_filled_ibins(self):  # return indices of bins with nonzero contents
        return [i for i, c in enumerate(self.bin_contents) if c > 0.]

    # ----------------------------------------------------------------------------------------
    def get_filled_bin_xbounds(self, extra_pads=0):  # low edge of lowest filled bin, high edge of highest filled bin (for search: "ignores empty bins")
        fbins = self.get_filled_ibins()
        if len(fbins) > 0:
            imin, imax = fbins[0], fbins[-1]
        else:
            imin, imax = 1, self.n_bins + 1  # not sure this is really the best default, but i'm adding it long after writing the rest of the fcn and it seems ok? (note after afterwards: it might be better to return None, but this is better than what i originally had which included under/overflows)
        if extra_pads > 0:  # give a little extra space on either side
            imin = max(0, imin - extra_pads)
            imax = min(len(self.low_edges) - 1, imax + extra_pads)
        return self.low_edges[imin], self.low_edges[imax + 1] if imax + 1 < len(self.low_edges) else self.xmax  # if it's not already the overflow bin, we want the next low edge, otherwise <self.xmax>

    # ----------------------------------------------------------------------------------------
    def get_bounds(self, include_overflows=False):
        if include_overflows:
            imin, imax = 0, self.n_bins + 2
        else:
            imin, imax = 1, self.n_bins + 1
        return imin, imax

    # ----------------------------------------------------------------------------------------
    def binwidth(self, ibin):
        if ibin == 0:  # use width of first bin for underflow
            ibin += 1
        elif ibin == self.n_bins + 1:  # and last bin for overflow
            ibin -= 1
        return self.low_edges[ibin+1] - self.low_edges[ibin]

    # ----------------------------------------------------------------------------------------
    def integral(self, include_overflows, ibounds=None, multiply_by_bin_width=False, multiply_by_bin_center=False):
        """ NOTE by default does not multiply by bin widths """
        if ibounds is None:
            imin, imax = self.get_bounds(include_overflows)
        else:
            imin, imax = ibounds
        sum_value = 0.0
        for ib in range(imin, imax):
            sum_value += self.bin_contents[ib] * (self.binwidth(ib) if multiply_by_bin_width else 1) * (self.get_bin_centers()[ib] if multiply_by_bin_center else 1)
        return sum_value

    # ----------------------------------------------------------------------------------------
    def normalize(self, include_overflows=True, expect_overflows=False, overflow_eps_to_ignore=1e-15, multiply_by_bin_width=False):
        sum_value = self.integral(include_overflows, multiply_by_bin_width=multiply_by_bin_width)
        if multiply_by_bin_width and any(abs(self.binwidth(i)-self.binwidth(1)) > eps for i in self.ibiniter(False)):
            print('  %s normalizing with multiply_by_bin_width set, but bins aren\'t all the same width, which may not work' % wrnstr())  # it would be easy to add but i don't want to test it now
        imin, imax = self.get_bounds(include_overflows)
        if sum_value == 0.0:
            return
        if not expect_overflows and not include_overflows and (self.bin_contents[0]/sum_value > overflow_eps_to_ignore or self.bin_contents[self.n_bins+1]/sum_value > overflow_eps_to_ignore):
            print('WARNING under/overflows in Hist::normalize()')
        for ib in range(imin, imax):
            self.bin_contents[ib] /= sum_value
            if self.sum_weights_squared is not None:
                self.sum_weights_squared[ib] /= sum_value*sum_value
            if self.errors is not None:
                self.errors[ib] /= sum_value
        check_sum = 0.0
        for ib in range(imin, imax):  # check it
            check_sum += self.bin_contents[ib] * (self.binwidth(ib) if multiply_by_bin_width else 1)
        if not is_normed(check_sum, this_eps=1e-10):
            raise Exception('not normalized: %f' % check_sum)
        self.ytitle = 'fraction of %.0f' % sum_value

    # ----------------------------------------------------------------------------------------
    def sample(self, n_vals, include_overflows=False, debug_plot=False):  # draw <n_vals> random numbers from the x axis, according to the probabilities given by the bin contents NOTE similarity to recombinator.choose_vdj_combo()
        assert not include_overflows  # probably doesn't really make sense (since contents of overflows could've been from anywhere below/above, but we'd only return bin center), this is just a way to remind that it doesn't make sense
        self.normalize(include_overflows=include_overflows)  # if this is going to get called a lot with n_vals of 1, this would be slow, but otoh we *really* want to make sure things are normalized with include_overflows the same as it is here
        centers = self.get_bin_centers()
        pvals = np.random.uniform(0, 1, size=n_vals)
        return_vals = [None for _ in pvals]
        sum_prob, last_sum_prob = 0., 0.
        for ibin in self.ibiniter(include_overflows):
            sum_prob += self.bin_contents[ibin]
            for iprob, pval in enumerate(pvals):
                if pval < sum_prob and pval >= last_sum_prob:
                    return_vals[iprob] = centers[ibin]
            last_sum_prob = sum_prob
        assert return_vals.count(None) == 0

        if debug_plot:
            from . import plotting
            fig, ax = plotting.mpl_init()
            self.mpl_plot(ax, label='original')
            shist = Hist(value_list=return_vals, init_int_bins=True)
            shist.normalize(include_overflows=False)
            shist.mpl_plot(ax, label='sampled', color='red')
            plotting.mpl_finish(ax, '', 'tmp')

        return return_vals

    # ----------------------------------------------------------------------------------------
    def logify(self, factor):
        for ib in self.ibiniter(include_overflows=True):
            if self.bin_contents[ib] > 0:
                if self.bin_contents[ib] <= factor:
                    raise Exception('factor %f passed to hist.logify() must be less than all non-zero bin entries, but found a bin with %f' % (factor, self.bin_contents[ib]))
                self.bin_contents[ib] = math.log(self.bin_contents[ib] / float(factor))
            if self.errors[ib] > 0:  # I'm not actually sure this makes sense
                self.errors[ib] = math.log(self.errors[ib] / float(factor))

    # ----------------------------------------------------------------------------------------
    def divide_by(self, denom_hist, debug=False):
        """ NOTE doesn't check bin edges are the same, only that they've got the same number of bins """
        if self.n_bins != denom_hist.n_bins or self.xmin != denom_hist.xmin or self.xmax != denom_hist.xmax:
            raise Exception('ERROR bad limits in Hist::divide_by')
        for ib in range(0, self.n_bins + 2):
            if debug:
                print(ib, self.bin_contents[ib], float(denom_hist.bin_contents[ib]))
            if denom_hist.bin_contents[ib] == 0.0:
                self.bin_contents[ib] = 0.0
            else:
                self.bin_contents[ib] /= float(denom_hist.bin_contents[ib])

    # ----------------------------------------------------------------------------------------
    # NOTE if you're here, you may be looking for plotting.make_mean_hist()
    def add(self, h2, debug=False):
        """ NOTE doesn't check bin edges are the same, only that they've got the same number of bins """
        if self.n_bins != h2.n_bins or self.xmin != h2.xmin or self.xmax != h2.xmax:
            raise Exception('ERROR bad limits in Hist::add')
        for ib in range(0, self.n_bins + 2):
            if debug:
                print(ib, self.bin_contents[ib], float(h2.bin_contents[ib]))
            self.bin_contents[ib] += h2.bin_contents[ib]

    # ----------------------------------------------------------------------------------------
    def write(self, outfname):
        if not os.path.exists(os.path.dirname(outfname)):
            os.makedirs(os.path.dirname(outfname))
        with open(outfname, csv_wmode()) as outfile:
            header = [ 'bin_low_edge', 'contents', 'binlabel' ]
            if self.errors is not None:
                header.append('error')
            else:
                header.append('sum-weights-squared')
            writer = csv.DictWriter(outfile, header)
            writer.writeheader()
            for ib in range(self.n_bins + 2):
                row = {'bin_low_edge':self.low_edges[ib], 'contents':self.bin_contents[ib], 'binlabel':self.bin_labels[ib] }
                if self.errors is not None:
                    row['error'] = self.errors[ib]
                else:
                    row['sum-weights-squared'] = self.sum_weights_squared[ib]
                writer.writerow(row)

    # ----------------------------------------------------------------------------------------
    def get_bin_centers(self, ignore_overflows=False):
        bin_centers = []
        for ibin in range(len(self.low_edges)):
            low_edge = self.low_edges[ibin]
            if ibin < len(self.low_edges) - 1:
                high_edge = self.low_edges[ibin + 1]
            else:
                high_edge = low_edge + (self.low_edges[ibin] - self.low_edges[ibin - 1])  # overflow bin has undefined upper limit, so just use the next-to-last bin width
            bin_centers.append(0.5 * (low_edge + high_edge))
        if ignore_overflows:
            return bin_centers[1:-1]
        else:
            return bin_centers

    # ----------------------------------------------------------------------------------------
    def bin_contents_no_zeros(self, value):
        """ replace any zeros with <value> """
        bin_contents_no_zeros = list(self.bin_contents)
        for ibin in range(len(bin_contents_no_zeros)):
            if bin_contents_no_zeros[ibin] == 0.:
                bin_contents_no_zeros[ibin] = value
        return bin_contents_no_zeros

    # ----------------------------------------------------------------------------------------
    def get_mean(self, ignore_overflows=False, absval=False, ibounds=None):
        if ibounds is not None:
            imin, imax = ibounds
            if self.integral(False, ibounds=(0, imin)) > 0 or self.integral(False, ibounds=(imax, self.n_bins + 2)) > 0:
                print('  %s called hist.get_mean() with ibounds %s that exclude bins with nonzero entries:  below %.3f   above %.3f' % (color('yellow', 'warning'), ibounds, self.integral(False, ibounds=(0, imin)), self.integral(False, ibounds=(imax, self.n_bins + 2))))
            if imin < 0:
                print('  %s increasing specified imin %d to 0' % (wrnstr(), imin))
                imin = 0
            if imax > self.n_bins + 2:
                print('  %s decreasing specified imax %d to %d' % (wrnstr(), imax, self.n_bins + 2))
                imax = self.n_bins + 2
        elif ignore_overflows:
            imin, imax = 1, self.n_bins + 1
        else:
            imin, imax = 0, self.n_bins + 2
        centers = self.get_bin_centers()
        total, integral = 0.0, 0.0
        for ib in range(imin, imax):
            total += self.bin_contents[ib] * (abs(centers[ib]) if absval else centers[ib])
            integral += self.bin_contents[ib]
        if integral > 0.:
            return total / integral
        else:
            return 0.

    # ----------------------------------------------------------------------------------------
    def rebin(self, factor):
        print('TODO implement Hist::rebin()')

    # ----------------------------------------------------------------------------------------
    def horizontal_print(self, bin_centers=False, bin_decimals=4, contents_decimals=3):
        bin_format_str = '%7.' + str(bin_decimals) + 'f'
        contents_format_str = '%7.' + str(contents_decimals) + 'f'
        binlist = self.get_bin_centers() if bin_centers else self.low_edges
        binline = ''.join([bin_format_str % b for b in binlist])
        contentsline = ''.join([contents_format_str % c for c in self.bin_contents])
        return [binline, contentsline]

    # ----------------------------------------------------------------------------------------
    def __str__(self, print_ibin=False):
        str_list = ['   %s %10s%12s%s'  % ('ibin ' if print_ibin else '', 'low edge', 'contents', '' if self.errors is None else '     err'), '\n', ]
        for ib in range(len(self.low_edges)):
            str_list += ['   %s %10.4f%12.3f'  % ('%4d'%ib if print_ibin else '', self.low_edges[ib], self.bin_contents[ib]), ]
            if self.errors is not None:
                str_list += ['%9.2f' % self.errors[ib]]
            if self.bin_labels.count('') != len(self.bin_labels):
                str_list += ['%12s' % self.bin_labels[ib]]
            if ib == 0:
                str_list += ['   (under)']
            if ib == len(self.low_edges) - 1:
                str_list += ['   (over)']
            str_list += ['\n']
        return ''.join(str_list)

    # ----------------------------------------------------------------------------------------
    # NOTE remove_empty_bins can be a bool (remove/not all empty bins) or a list of length two (remove empty bins outside range)
    def mpl_plot(self, ax, ignore_overflows=False, label=None, color=None, alpha=None, linewidth=None, linestyle=None, markersize=None, errors=True, remove_empty_bins=False,
                 square_bins=False, no_vertical_bin_lines=False):
        # ----------------------------------------------------------------------------------------
        def keep_bin(xv, yv):
            if isinstance(remove_empty_bins, list):
                xmin, xmax = remove_empty_bins
                if xv > xmin and xv < xmax:
                    return True  # always keep within range
            return yv != 0.
        # ----------------------------------------------------------------------------------------
        def sqbplot(kwargs):
            kwargs['markersize'] = 0
            for ibin in self.ibiniter(include_overflows=False):
                if not keep_bin(self.low_edges[ibin], self.bin_contents[ibin]):  # maybe should use bin centers rather than low edges, i dunno
                    continue
                tplt = ax.plot([self.low_edges[ibin], self.low_edges[ibin+1]], [self.bin_contents[ibin], self.bin_contents[ibin]], **kwargs)  # horizontal line for this bin
                kwargs['label'] = None  # make sure there's only one legend entry for each hist
                if not no_vertical_bin_lines:
                    ax.plot([self.low_edges[ibin], self.low_edges[ibin]], [self.bin_contents[ibin-1], self.bin_contents[ibin]], **kwargs)  # vertical line from last bin contents
                if errors:
                    bcenter = self.get_bin_centers()[ibin]
                    tplt = ax.plot([bcenter, bcenter], [self.bin_contents[ibin] - self.errors[ibin], self.bin_contents[ibin] + self.errors[ibin]], **kwargs)  # horizontal line for this bin
            if not no_vertical_bin_lines:
                tplt = ax.plot([self.low_edges[ibin+1], self.low_edges[ibin+1]], [self.bin_contents[ibin], 0], **kwargs)  # vertical line for right side of last bin
            return tplt  # not sure if this gets used anywhere?
        # ----------------------------------------------------------------------------------------
        # UPDATE maybe it can handle floats now so i don't need this?
        # if linewidth is not None:  # i'd really rather this wasn't here, but the error message mpl kicks is spectacularly uninformative so you have to catch it beforehand (when writing the svg file, it throws TypeError: Cannot cast array data from dtype('<U1') to dtype('float64') according to the rule 'safe')
        #     if not isinstance(linewidth, int):
        #         raise Exception('have to pass linewidth as int, not %s' % type(linewidth))
        # note: bin labels are/have to be handled elsewhere
        if self.integral(include_overflows=(not ignore_overflows)) == 0.0:
            # print '   integral is zero in hist::mpl_plot'
            return None
        if ignore_overflows:
            xvals = self.get_bin_centers()[1:-1]
            yvals = self.bin_contents[1:-1]
            yerrs = self.errors[1:-1]
        else:
            xvals = self.get_bin_centers()
            yvals = self.bin_contents
            yerrs = self.errors

        defaults = {'color' : 'black',
                    'alpha' : 0.6,
                    'linewidth' : 3,
                    'linestyle' : '-',
                    'marker' : '.',
                    'markersize' : 13}
        kwargs = {}
        argvars = locals()
        for arg in defaults:
            if arg in argvars and argvars[arg] is not None:
                kwargs[arg] = argvars[arg]
            else:
                kwargs[arg] = defaults[arg]
        if label is not None:
            kwargs['label'] = label
        elif self.title != '':
            kwargs['label'] = self.title
        if remove_empty_bins is not False:  # NOTE can be bool, but can also be list of length two (remove bins only outside those bounds)
            xvals, yvals, yerrs = zip(*[(xvals[iv], yvals[iv], yerrs[iv]) for iv in range(len(xvals)) if keep_bin(xvals[iv], yvals[iv])])
        if errors and not square_bins:
            kwargs['yerr'] = yerrs
            return ax.errorbar(xvals, yvals, **kwargs)  #, fmt='-o')
        else:
            if square_bins:
                return sqbplot(kwargs)
            else:
                return ax.plot(xvals, yvals, **kwargs)  #, fmt='-o')

    # ----------------------------------------------------------------------------------------
    def fullplot(self, plotdir, plotname, pargs={}, fargs={}, texts=None, only_csv=False): #**kwargs):  # i.e. full plotting process, not just the ax.plot type stuff above
        self.write('%s/%s.csv'%(plotdir, plotname))
        if only_csv:
            return
        from . import plotting
        fig, ax = plotting.mpl_init()  # this'll need to be updated when i want to use a kwarg for this fcn
        self.mpl_plot(ax, **pargs)
        if texts is not None:
            for xv, yv, tx in texts:
                fig.text(xv, yv, tx, fontsize=15)
        if 'xticks' not in fargs and any(l != '' for l in self.bin_labels):
            fargs['xticks'] = self.get_bin_centers()
            fargs['xticklabels'] = self.bin_labels
        return plotting.mpl_finish(ax, plotdir, plotname, **fargs)

# ----------------------------------------------------------------------------------------
def make_mean_hist(hists, ignore_empty_bins=False, percentile_err=None):
    """ return the hist with bin contents the mean over <hists> of each bin """
    binvals = {}
    for hist in hists:  # I could probably do this with list comprehensions or something, but this way handles different bin bounds
        for ib in range(0, hist.n_bins + 2):
            low_edge = hist.low_edges[ib]
            if low_edge not in binvals:
                binvals[low_edge] = []
            binvals[low_edge].append(hist.bin_contents[ib])
    binlist = sorted(binvals.keys())
    meanhist = Hist(len(binlist) - 2, binlist[1], binlist[-1], xbins=binlist[1 :])
    for ib in range(len(binlist)):
        vlist = binvals[binlist[ib]]
        if ignore_empty_bins:
            vlist = [v for v in vlist if v > 0]
        if len(vlist) == 0:
            continue
        if percentile_err is None:
            err = 0 if len(vlist)==1 else (np.std(vlist, ddof=1) / math.sqrt(len(vlist)))
        else:
            err = vlist[0] if len(vlist)==1 else 0.5*(np.percentile(vlist, 67) - np.percentile(vlist, 33))
        meanhist.set_ibin(ib, np.mean(vlist), error=err)
    # meanhist.normalize()
    return meanhist


# fmt: on
