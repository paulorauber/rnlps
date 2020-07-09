"""

Creates config files for experiments and hyperparamter grid search.

"""

import os
import argparse
import json
import numpy as np
from itertools import product
from collections import namedtuple

# Configuration templates with different arguments
CfgOracle = namedtuple('Oracle', [])

CfgRandom = namedtuple('Random', ['seed'])

# Recurrent neural-linear
CfgThompsonRNN = namedtuple('ThompsonRecurrentNetwork',
                               ['n_units', 'learning_rate', 'regularise_lambda', 'epochs',
                                'train_every', 'std_targets', 'std_weights',
                                'verbose', 'seed'])


# Neural-linear with sinusoidal units
CfgThompsonSinNN = namedtuple('ThompsonSinFeedforwardNetwork',
                           ['order', 'periods', 'periods_dim', 'n_units',
                            'learning_rate', 'regularise_lambda','epochs', 'train_every',
                            'std_targets', 'std_weights', 'verbose', 'seed'])

def main():
    bandit = "StationaryContextualBandit"
    bandit_parameters = {"dataset": "wall_following_24", "seed": 0}

    # Number of interactions (has to be lesser than the size of dataset)
    trial_length = 5455

    # Policy settings: Defines the hyperparameter grid.
    seeds = list(range(5))

    configs = []

    # Policy settings (Oracle)
    configs.append(CfgOracle())

    # Policy settings (Random)
    configs.append(CfgRandom(seed=seeds))

    # Policy settings grid (Recurrent neural-linear)
    configs.append(CfgThompsonRNN(n_units=[[32, 32, 32]],
                                  learning_rate=[0.001, 0.01, 0.1],
                                  regularise_lambda=[0.001],
                                  epochs=[16, 64],
                                  train_every=[32, 128],
                                  std_targets=[0.3],
                                  std_weights=[0.5, 1],
                                  verbose=[False],
                                  seed=seeds))

    # Policy settings grid (Neural-linear)
    configs.append(CfgThompsonSinNN(order=[1,4],
                                 periods=[[]],
                                 periods_dim =[2,4,8],
                                 n_units= [[64, 64, 64]],
                                 learning_rate=[0.001, 0.01, 0.1],
                                 regularise_lambda=[0.001],
                                 epochs=[16, 64],
                                 train_every=[32, 128],
                                 std_targets=[0.1, 0.3],
                                 std_weights=[0.5, 1],
                                 verbose=[False],
                                 seed=seeds))

    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Experiments directory.')
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        os.makedirs(args.directory)

    # Creates folders with config files for all combinations of hyperparameters.

    i = 1
    for policy_config in configs:
        PolicyConfig = type(policy_config)
        combos = map(lambda c: PolicyConfig(*c), product(*policy_config))

        for combo in combos:
            print(combo)

            cfg = {'bandit': bandit,
                   'bandit_parameters': bandit_parameters,
                   'trial_length': trial_length,
                   'policy': PolicyConfig.__name__,
                   'policy_parameters': combo._asdict()}

            path = os.path.join(args.directory, str(i))
            if not os.path.isdir(path):
                os.makedirs(path)

            f = open(os.path.join(path, 'config.json'), 'w')
            json.dump(cfg, f)
            f.close()

            i += 1


if __name__ == "__main__":
    main()
