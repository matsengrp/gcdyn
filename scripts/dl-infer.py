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
from gcdyn.models import NeuralNetworkModel
from gcdyn.poisson import ConstantResponse

# ----------------------------------------------------------------------------------------
def get_df(smpl, result):
    df = pd.DataFrame(
        {
            'Predicted': np.array([row[0].value for row in result]),
            'Truth': np.array(
                [row[0].xscale for row in smpldict[smpl]['responses']], dtype=str
            ),
        }
    )
    ofn = '%s/%s.csv' % (args.outdir, smpl)
    sns.histplot(df, x='Predicted', hue='Truth', binwidth=0.025)
    df.to_csv(ofn)
    plt.savefig('%s/%s-hist.svg' % (args.outdir, smpl))
    return df

# ----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('infname', help='input merge simulation dill pickle file')
parser.add_argument('outdir')
parser.add_argument('--epochs', default=100)
parser.add_argument('--test', action='store_true', help='sets things to be super fast, so not useful for real inference, but just to check if things are running properly')

start = time.time()
args = parser.parse_args()
if args.test:
    args.epochs = 10

# ----------------------------------------------------------------------------------------
with open(args.infname, 'rb') as f:
    samples = pickle.load(f)

N = len(samples['trees'])

sublist = lambda x, idx: [x[i] for i in idx]

idxs, smpldict = {}, {}
idxs['train'] = random.sample(range(N), int(0.8 * N))
idxs['test'] = [i for i in range(N) if i not in idxs['train']]
smpldict['train'] = {key: sublist(val, idxs['train']) for key, val in samples.items()}
smpldict['test'] = {key: sublist(val, idxs['test']) for key, val in samples.items()}

param_to_predict = [
    [ConstantResponse(row[0].xscale)] for row in smpldict['train']['responses']
]

model = NeuralNetworkModel(smpldict['train']['trees'], param_to_predict)
model.fit(epochs=args.epochs)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
print('  writing train/test results to %s' % args.outdir)

result = model.predict(smpldict['train']['trees'], ladderize_trees=False)
get_df('train', result)
result = model.predict(smpldict['test']['trees'])
get_df('test', result)

print('    total dl inference time: %.1f sec' % (time.time()-start))
