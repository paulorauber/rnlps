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

CfgSWLinUCB = namedtuple('SW_LinUCB', ['delta', 'alpha', 'tau', 'lambda_reg', 'sigma_noise', 'seed'])

CfgDLinUCB = namedtuple('D_LinUCB', ['delta', 'alpha', 'gamma', 'lambda_reg', 'sigma_noise', 'seed'])

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
    # Bandit settings - problem type and specific instance.
    bandit = "RotatingLinearBandit2d"
    bandit_parameters = {"n_arms": 25, "time_period": 32, "seed": 0}

    trial_length = 4096

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
                                  std_targets=[0.1, 0.3],
                                  std_weights=[0.5, 1],
                                  verbose=[False],
                                  seed=seeds))

    # Policy settings grid (Neural-linear)
    configs.append(CfgThompsonSinNN(order=[1, 4],
                                 periods=[[]],
                                 periods_dim =[2,4,8],
                                 n_units= [[32, 32, 32], [64, 64, 64]],
                                 learning_rate=[0.001, 0.01, 0.1],
                                 regularise_lambda=[0.001],
                                 epochs=[16, 64],
                                 train_every=[32, 128],
                                 std_targets=[0.1, 0.3],
                                 std_weights=[0.5, 1],
                                 verbose=[False],
                                 seed=seeds))


    configs.append(CfgSWLinUCB(delta = [0.1],
                             alpha = [1],
                             tau = [128, 256, 512, 1024],
                             lambda_reg = [0.1],
                             sigma_noise = [0.05],
                             seed=seeds))

    configs.append(CfgDLinUCB(delta = [0.1],
                             alpha = [1],
                             gamma = [0.9, 0.95, 0.98, 0.99],
                             lambda_reg = [0.1],
                             sigma_noise = [0.05],
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
