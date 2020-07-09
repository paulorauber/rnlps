"""
    Runs a single experiment for a particular configuration of bandit and
    policy settings. Saves results in trial.csv. 

"""

import os
import argparse
import json
import numpy as np
import pandas as pd

from rnlps.non_contextual_bandits import non_contextual_bandits
from rnlps.contextual_bandits import contextual_bandits
from rnlps.linear_bandits import linear_bandits

from rnlps.non_contextual_policies import non_contextual_policies
from rnlps.contextual_policies import contextual_policies
from rnlps.contextual_linear_policies import contextual_linear_policies

def main():
    """ Example configuration file (also see the output of hgrid.py):
    {
    "bandit": "StationaryBernoulliBandit",
    "bandit_parameters": {"means": [0.25, 0.5, 0.75], "seed": 0},

    "policy": "ThompsonSamplingBernoulli",
    "policy_parameters": {"a": 1.0, "b": 1.0, "seed": 1},

    "trial_length": 100
    }
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Experiment directory.')
    args = parser.parse_args()

    f = open(os.path.join(args.directory, 'config.json'), 'r')
    config = json.load(f)
    f.close()

    if config['bandit'] in non_contextual_bandits.keys():
        bandit = non_contextual_bandits[config['bandit']](**config['bandit_parameters'])
        policy = non_contextual_policies[config['policy']](bandit, **config['policy_parameters'])
    elif config['bandit'] in contextual_bandits.keys():
        bandit = contextual_bandits[config['bandit']](**config['bandit_parameters'])
        policy = contextual_policies[config['policy']](bandit, **config['policy_parameters'])
    else:
        bandit = linear_bandits[config['bandit']](**config['bandit_parameters'])
        policy = contextual_linear_policies[config['policy']](bandit, **config['policy_parameters'])

    trial_length = config['trial_length']

    trial = policy.interact(trial_length)

    df = pd.DataFrame({'Pull': np.arange(trial_length) + 1,
                       'Return': trial.cumulative_rewards(),
                       'Arm_Pulled': trial.arms,
                       'Regret': trial.cumulative_regret(),
                       'Policy': repr(policy)})

    df.to_csv(os.path.join(args.directory, 'trial.csv'), index=False)


if __name__ == '__main__':
    main()
