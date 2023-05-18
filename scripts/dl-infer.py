#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from Bio import SeqIO
# import colored_traceback.always  # need to add this to installation stuff, i'm not sure how to do it atm
import dill
import time
import copy
import pickle
import pandas as pd
import seaborn as sns
import random

# if you move this script, you'll need to change this method of getting the imports
gcd_dir = os.path.dirname(os.path.realpath(__file__)).replace('/scripts', '')
sys.path.insert(1, gcd_dir + '/gcdyn')

from colors import color

# ----------------------------------------------------------------------------------------
def csvfn(smpl):
    return '%s/%s.csv' % (args.outdir, smpl)

# ----------------------------------------------------------------------------------------
def make_plot(smpl, df):
    sns.color_palette("hls", 8)
    plt.clf()
    sns.histplot(df, x='Predicted', hue='Truth', bins=30, multiple='stack') #.set(title=smpl) #binwidth=0.025)
    df.to_csv(csvfn(smpl))
    plt.savefig('%s/%s-hist.svg' % (args.outdir, smpl))

# ----------------------------------------------------------------------------------------
def get_df(smpl, result, smpldict):
    df = pd.DataFrame(
        {
            'Predicted': np.array([row[0].value for row in result]),
            'Truth': np.array(
                [row[0].xscale for row in smpldict[smpl]['responses']], dtype=str
            ),
        }
    )
    make_plot(smpl, df)
    return df

# ----------------------------------------------------------------------------------------
def read_plot_csv():
    for smpl in ['train', 'test']:
        df = pd.read_csv(csvfn(smpl))
        make_plot(smpl, df)

# ----------------------------------------------------------------------------------------
def train_and_test():
    from gcdyn.models import NeuralNetworkModel
    from gcdyn.poisson import ConstantResponse

    with open(args.infname, 'rb') as f:
        samples = pickle.load(f)  # dict with two keys ('trees' and 'responses'), and a list for each
    print('    read %d trees and %d responses from %s'% (len(samples['trees']), len(samples['responses']), args.infname))
    print('      first response pair:\n        birth: %s\n        death: %s' % (samples['responses'][0][0], samples['responses'][0][1]))

    n_trees = len(samples['trees'])
    idxs, smpldict = {}, {}
    idxs['train'] = random.sample(range(n_trees), int(args.train_frac * n_trees))
    idxs['test'] = [i for i in range(n_trees) if i not in idxs['train']]

    sublist = lambda x, idx: [x[i] for i in idx]
    for smpl in ['train', 'test']:
        smpldict[smpl] = {key: [val[i] for i in idxs[smpl]] for key, val in samples.items()}

    param_to_predict = [
        [ConstantResponse(birth_resp.xscale)] for birth_resp, death_resp in smpldict['train']['responses']
    ]

    model = NeuralNetworkModel(smpldict['train']['trees'], param_to_predict)
    model.fit(epochs=args.epochs)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    print('  writing train/test results to %s' % args.outdir)

    result = model.predict(smpldict['train']['trees'], ladderize_trees=False)
    get_df('train', result, smpldict)
    result = model.predict(smpldict['test']['trees'])
    get_df('test', result, smpldict)

    print('    total dl inference time: %.1f sec' % (time.time()-start))

# ----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('infname', help='input merge simulation dill pickle file')
parser.add_argument('outdir')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--train-frac', type=float, default=0.8, help='train on this fraction of the trees')
parser.add_argument('--test', action='store_true', help='sets things to be super fast, so not useful for real inference, but just to check if things are running properly')
parser.add_argument('--overwrite', action='store_true')

start = time.time()
args = parser.parse_args()
if args.test:
    args.epochs = 10

# ----------------------------------------------------------------------------------------
if os.path.exists(csvfn('test')) and not args.overwrite:
    print('    csv files already exist, so just replotting (override with --overwrite): %s' % csvfn('test'))
    read_plot_csv()
    sys.exit(0)

train_and_test()
