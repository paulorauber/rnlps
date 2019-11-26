""" Creates a csv file summarising the performance of all the policies
    on an experiment. """

import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('directory', help='Experiments directory.')
args = parser.parse_args()
currd = os.getcwd()
os.chdir(args.directory)
folders = os.listdir()
folders = [f for f in folders if f.isdigit()]
d2 = pd.DataFrame(columns=['Policy','Pull','Return'])
for f in folders:
    exists = os.path.isfile(f + "/trial.csv")
    if exists:
        df = pd.read_csv(f + "/trial.csv")
        r = df.iloc[-1,:]
        r['folder'] = f
        d2 = d2.append(r)

d2 = d2.sort_values(by='Return', ascending = False)
d2 = d2[['Policy', 'Pull', 'Return', 'folder']]
d2.to_csv("Analysis.csv")

aggregations = {'Return':{'Mean_Return': 'mean', 'Std_Return': 'std',
                'Num_experiments': 'count'}}

d3 = d2.groupby('Policy').agg(aggregations)
d3.columns = d3.columns.droplevel()
d3 = d3.sort_values(by='Mean_Return', ascending=False)

# Save csv file in the 'Experiments directory' which was given as an argument.
d3.to_csv("policy_mean_perf.csv")

os.chdir(currd)
