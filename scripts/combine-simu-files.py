import argparse
import os
import numpy as np

# import colored_traceback.always  # need to add this to installation stuff, i'm not sure how to do it atm
import dill

# ----------------------------------------------------------------------------------------
ustr = """
Read encoded trees and response functions from list of input dirs, and combine (concatenate) them into one output file
"""
parser = argparse.ArgumentParser(usage=ustr)
parser.add_argument(
    "indirs", nargs="+", help="list of dirs with gcdyn simulation files"
)
parser.add_argument(
    "--outdir",
    required=True,
    help="output dir name for concatenated simulation info (response functions and trees)",
)
args = parser.parse_args()

def tree_fn(idr):
    return '%s/encoded-trees.npy' % idr
def resp_fn(idr):
    return '%s/responses.pkl' % idr

encd_trees, responses = [], []
for idr in args.indirs:
    for etree in np.load(tree_fn(idr)):
        encd_trees.append(etree)
    with open(resp_fn(idr), "rb") as rfile:
        responses += dill.load(rfile)

print(
    "  writing %d trees from %d files to %s"
    % (len(encd_trees), len(args.indirs), args.outdir)
)
if not os.path.exists(os.path.dirname(args.outdir)):
    os.makedirs(os.path.dirname(args.outdir))
with open(resp_fn(args.outdir), "wb") as rfile:
    dill.dump(responses, rfile)
np.save(tree_fn(args.outdir), encd_trees)
