r"""Utility functions ^^^^^^^^^^^^^^^^^"""

from collections import defaultdict

import ete3
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform
import seaborn as sns
import platform
import resource
import psutil
import os
import glob


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
    if (
        node.chain_2_start_idx is None
    ):  # default is set in bdms.TreeNode, indicates single chain
        return simple_fivemer_contexts(node.sequence)
    else:
        return padded_fivemer_contexts_of_paired_sequences(
            node.sequence, node.chain_2_start_idx
        )


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


def ladderize_tree(tree, attr="x"):
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

    sort_criteria = defaultdict(list)

    for node in tree.traverse("postorder"):
        if node.is_leaf():
            sort_criteria[node.name] = [node.t, node.up.t, getattr(node, attr)]
        else:
            sort_criteria[node.name] = sorted(
                (sort_criteria[child.name] for child in node.children), reverse=True
            )[0]

    for node in tree.traverse("postorder"):
        if len(node.children) > 1:
            node.children = sorted(
                node.children,
                key=lambda node: sort_criteria[node.name],
                reverse=True,
            )


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
# plot scatter + box/whisker plot comparing true and predicted values for deep learning inference
# NOTE leaving some commented code that makes plots we've been using recently, since we're not sure which plots we'll end up wanting in the end (and what's here is very unlikely to stay for very long)
def make_dl_plot(smpl, df, outdir):
    plt.clf()
    sns.set_palette("viridis", 8)
    # hord = sorted(set(df["Truth"]))
    # sns.histplot(df, x="Predicted", hue="Truth", hue_order=hord, palette="tab10", bins=30, multiple="stack", ).set(title=smpl)
    ax = sns.boxplot(
        df,
        x="Truth",
        y="Predicted",
        boxprops={"facecolor": "None"},
        order=sorted(set(df["Truth"])),
    )
    if len(df) < 2000:
        ax = sns.swarmplot(df, x="Truth", y="Predicted", size=4, alpha=0.6)
    ax.set(title=smpl)
    for xv, xvl in zip(ax.get_xticks(), ax.get_xticklabels()):
        plt.plot(
            [xv - 0.5, xv + 0.5],
            [float(xvl._text), float(xvl._text)],
            color="darkred",
            linestyle="--",
            linewidth=3,
            alpha=0.7,
        )
    # sns.scatterplot(df, x='Truth', y='Predicted')
    # xvals, yvals = df['Truth'], df['Predicted']
    # plt.plot([0.95 * min(xvals), 1.05 * max(xvals)], [0.95 * min(yvals), 1.05 * max(yvals)], color='darkred', linestyle='--', linewidth=3, alpha=0.7)
    plt.savefig("%s/%s-hist.svg" % (outdir, smpl))


# ----------------------------------------------------------------------------------------
def plot_trees(plotdir, pfo_list, xmin=-5, xmax=5, nsteps=40, n_to_plot=10):
    # ----------------------------------------------------------------------------------------
    def plt_single_tree(itree, pfo, xmin, xmax, n_bins=30):
        plt.clf()
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        leaves = list(pfo["tree"].iter_leaves())
        leaf_vals = [n.x for n in leaves]
        int_vals = [n.x for n in pfo["tree"].iter_descendants() if n not in leaves]
        all_vals = leaf_vals + int_vals
        xmin, xmax = min([xmin] + all_vals), max([xmax] + all_vals)
        sns.histplot(
            {"internal": int_vals, "leaves": leaf_vals},
            ax=ax2,
            multiple="stack",
            binwidth=(xmax - xmin) / n_bins,
        )

        dx = (xmax - xmin) / nsteps
        xvals = list(np.arange(xmin, 0, dx)) + list(np.arange(0, xmax + dx, dx))
        rvals = [pfo["birth-response"].λ_phenotype(x) for x in xvals]
        data = {"affinity": xvals, "lambda": rvals}
        sns.lineplot(
            data, x="affinity", y="lambda", ax=ax, linewidth=3, color="#990012"
        )
        ax.set(
            title="xscale %.1f  xshift %.1f (%d total nodes, %d leaves)"
            % (
                pfo["birth-response"].xscale,
                pfo["birth-response"].xshift,
                len(all_vals),
                len(leaf_vals),
            )
        )
        plt.savefig("%s/trees-%d.svg" % (plotdir, itree))

    # ----------------------------------------------------------------------------------------
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    for itree, pfo in enumerate(pfo_list[:n_to_plot]):
        plt_single_tree(itree, pfo, xmin, xmax)
    make_html(plotdir, n_columns=4)
    print("    plotting trees to %s" % plotdir)


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
