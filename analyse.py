""" Visualisaton - Plots regret/return performance of policies. """

import matplotlib
matplotlib.use('Agg')
import os
import argparse
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rnlps.bandits import bandits


def main():

    # set include_default_policies to True if you want to include default
    # policies in the plot.
    include_default_policies = True

    # Modify this if using a different default.
    default_rnn_policy = 'ThompsonRecurrentNetwork(n_units=[16, 16, 16], learning_rate=0.01, regL2=0.001, epochs=16, train_every=32, std_targets=0.3, std_weights=1.0)'
    default_ffnn_policy = 'ThompsonFeedforwardNetwork(order=1, periods=[ 2  4  8 16 32], n_units=[32, 32, 32], learning_rate=0.01, regL2=0.001, epochs=64, train_every=32, std_targets=0.3, std_weights=1.0)'

    # set separate_periods to True if you want separate plots for best FFNN
    # with and without periods
    separate_periods = True

    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Experiments directory.')
    parser.add_argument('--regret', help='Regret analysis.',
                        action='store_true')
    parser.add_argument('--reward', help='Reward analysis.',
                        action='store_true')
    args = parser.parse_args()

    dirs = [os.path.join(args.directory, d)
            for d in os.listdir(args.directory) if not d.startswith('_')]
    dirs = [d for d in dirs if os.path.isdir(d)]

    configs = {}
    for d in dirs:
        f = open(os.path.join(d, 'config.json'), 'r')
        configs[d] = json.load(f)
        f.close()

    bandit_settings = ['bandit', 'bandit_parameters', 'trial_length']
    reference = configs[dirs[0]]
    for d, config in configs.items():
        for p in bandit_settings:
            if config[p] != reference[p]:
                raise Exception('Inconsistent bandits ({0}).'.format(d))

    if args.regret:
        bandit = bandits[reference['bandit']](**reference['bandit_parameters'])
        ecr = bandit.expected_cumulative_rewards(reference['trial_length'])

    df = pd.DataFrame()
    for d in dirs:
        if os.path.exists(os.path.join(d, 'trial.csv')):
            frame = pd.read_csv(os.path.join(d, 'trial.csv'))
            frame = frame.sort_values(by='Pull')

            if args.regret:
                frame['Regret'] = ecr - frame['Return']
            if args.reward:
                first = frame['Return'][0]
                frame['Reward'] = np.ediff1d(frame['Return'], to_begin=first)

            df = df.append(frame)
        else:
            msg = 'Warning: missing trial {0} for {1}({2}).'
            print(msg.format(d, configs[d]['policy'],
                             configs[d]['policy_parameters']))


    last = df[df['Pull'] == reference['trial_length']]
    last = last.loc[:, ['Return', 'Policy']]

    best = last.groupby('Policy').mean().sort_values(by='Return',
                                                     ascending=False)

    with pd.option_context('display.max_colwidth', -1):
        print(best)

    best_ffnn = best[best.index.str.contains('ThompsonFeedforwardNetwork')]
    best_rnn = best[best.index.str.contains('ThompsonRecurrentNetwork')]
    if separate_periods:
        best_ffnn_with_p = best_ffnn[best_ffnn.index.str.
                                        contains('2  4  8 16 32')]
        best_ffnn_p_policy = best_ffnn_with_p.Return.idxmax()

    best_ffnn_policy = best_ffnn.Return.idxmax()
    best_rnn_policy = best_rnn.Return.idxmax()

    list_policies = ['Random()', 'ThompsonSamplingBernoulli(a=1, b=1)']

    list_policies.append(best_ffnn_policy)
    list_policies.append(best_rnn_policy)

    if include_default_policies:
        list_policies.append(default_ffnn_policy)
        list_policies.append(default_rnn_policy)

    print(best_rnn_policy)
    print(best_ffnn_policy)
    if separate_periods:
        print(best_ffnn_p_policy)
        list_policies.append(best_ffnn_p_policy)

    # Keep consistent colors
    c_list = sns.color_palette()
    c_palette = {'Random' : c_list[7],
     'Best RNN': c_list[0],
     'Best NN': c_list[1],
     'Default RNN': c_list[2],
     'Default NN': c_list[3],
     'Best NN (empty list of periods)':c_list[4],
     'Thompson Sampling (a = 1, b = 1)': c_list[5]}

    plot_df = df[df['Policy'].isin(list_policies)]
    plot_df['Policy_newnames'] = ''
    plot_df.loc[plot_df.Policy.str.contains('ThompsonRecurrentNetwork'), 'Policy_newnames'] = 'Best RNN'
    plot_df.loc[plot_df.Policy.str.contains('ThompsonFeedforwardNetwork'), 'Policy_newnames'] = 'Best NN (empty list of periods)'
    plot_df.loc[plot_df.Policy.str.contains('8 16 32'), 'Policy_newnames'] = 'Best NN'

    plot_df.loc[plot_df.Policy.str.contains('Random'), 'Policy_newnames'] = 'Random'

    plot_df.loc[plot_df.Policy.str.contains('ThompsonSamplingBernoulli'), 'Policy_newnames'] = 'Thompson Sampling (a = 1, b = 1)'
    if include_default_policies:
        plot_df.loc[plot_df.Policy == default_rnn_policy, 'Policy_newnames'] = 'Default RNN'
        plot_df.loc[plot_df.Policy == default_ffnn_policy, 'Policy_newnames'] = 'Default NN'

    plot_df = plot_df.sort_values(by='Policy_newnames')
    del plot_df['Policy']
    plot_df.rename(columns = {'Policy_newnames':'Policy'}, inplace=True)

    plot_df.to_csv('Analyse_plot_df.csv')
    plt.figure(figsize=(16,9))

    sns.set(context='paper', style='darkgrid', font_scale=3, rc={'legend.frameon':False, 'lines.linewidth':6.0})

    if args.regret:
        y = 'Regret'
    elif args.reward:
        y = 'Reward'
    else:
        y = 'Return'

    plot_df = plot_df[plot_df.Pull < int(reference['trial_length'])]

    # ci = "sd" draws the confidence interval using the standard deviation
    # set to "None" in order to have bootstrapped confidence intervals

    ax = sns.lineplot(x='Pull', y=y, hue='Policy', palette = c_palette,
                      data=plot_df, linewidth=3.0, ci="sd")

    plt.xlim(1, int(reference['trial_length']))
    plt.xticks(range(0, int(reference['trial_length']) + 1, 512))
    plt.xlabel('time step')
    plt.ylabel(y.lower())

    plt.savefig('res_sd.pdf',  bbox_inches='tight',pad_inches = 0)


if __name__ == "__main__":
    main()
