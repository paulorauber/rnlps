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

CfgSW_UCB = namedtuple('SW_UCB', ['tau','ksi','seed'])

CfgD_UCB = namedtuple('D_UCB', ['gamma','ksi','seed'])

# Recurrent neural-linear
CfgThompsonRNN = namedtuple('ThompsonRecurrentNetwork',
                               ['n_units', 'learning_rate', 'regularise_lambda', 'epochs',
                                'train_every', 'std_targets', 'std_weights',
                                'verbose', 'seed'])

# Neural-linear
CfgThompsonSinNN = namedtuple('ThompsonSinFeedforwardNetwork',
                           ['order', 'periods', 'periods_dim', 'n_units',
                            'learning_rate', 'regularise_lambda','epochs', 'train_every',
                            'std_targets', 'std_weights', 'verbose', 'seed'])

def main():
    # Bandit settings - problem type and specific instance.
    bandit = "SinusoidalBernoulliBandit"
    bandit_parameters = {"n_arms": 5, "step_size": (2*np.pi/32), "seed": 0}

    # Number of interactions
    trial_length = 4096

    # Policy settings: Defines the hyperparameter grid.
    seeds = list(range(3))

    configs = []

    # Policy settings (Oracle)
    configs.append(CfgOracle())

    # Policy settings (Random)
    configs.append(CfgRandom(seed=seeds))

    # Policy settings grid (Recurrent neural-linear)
    configs.append(CfgThompsonRNN(n_units=[[32, 32, 32]],
                                  learning_rate=[0.01],
                                  regularise_lambda=[0.001],
                                  epochs=[16],
                                  train_every=[32],
                                  std_targets=[0.1],
                                  std_weights=[0.5],
                                  verbose=[False],
                                  seed=seeds))

    # Policy settings grid (Neural-linear)
    configs.append(CfgThompsonSinNN(order=[1],
                                 periods=[[]],
                                 periods_dim = [1],
                                 n_units= [[32, 32, 32]],
                                 learning_rate=[0.1],
                                 regularise_lambda=[0.001],
                                 epochs=[16],
                                 train_every=[32],
                                 std_targets=[0.1],
                                 std_weights=[1.0],
                                 verbose=[False],
                                 seed=seeds))

    # Add SW_UCB & D_UCB

    configs.append(CfgSW_UCB(tau=[50, 75, 100],
                                  ksi=[0.5],
                                  seed=seeds))

    configs.append(CfgD_UCB(gamma = [0.99, 0.995, 0.999],
                                  ksi=[0.5],
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
