"""
    Generates the regret plot for an experiment. Includes the regret curves for
    the random policy, conventional algorithms like SW-UCB, and the default and
    best neural bandits.

    Usage:

    $ python3 regret_analysis.py experiment_folder/

    With no additional flag this takes the 'best' rnn and ffnn policies from the
    given_best_rnn_policy and given_best_ffnn_policy variables in this script.

    $ python3 regret_analysis.py experiment_folder/ --computebest

    Chooses the (R)NN policy with the highest return in this folder as 'best'
    for the plot.

    $ python3 regret_analysis.py experiment_folder/ --nondefaultasbest

    Assumes that the other non-default configuration in this folder is the
    'best'.

"""

import matplotlib
matplotlib.use('Agg')
import os
import argparse
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rnlps.non_contextual_bandits import non_contextual_bandits
from rnlps.contextual_bandits import contextual_bandits
from rnlps.linear_bandits import linear_bandits

sns.set(context='paper', style='darkgrid', font_scale=3, rc={'legend.frameon':False, 'lines.linewidth':6.0})

# Modify these if using different defaults

contextual_default_ffnn = "ThompsonSinFeedforwardNetwork(order=1,periods=[],periods_dim=2n_units=[32, 32, 32],learning_rate=0.01,regL2=0.001,epochs=64,train_every=32,std_targets=0.1,std_weights=1.0)"
contextual_default_rnn = "ThompsonRecurrentNetwork(n_units=[32,32,32],learning_rate=0.001,epochs=64,train_every=32,std_targets=0.3,std_weights=0.5,regL2=0.001)"

non_contextual_default_ffnn = "ThompsonFeedforwardNetwork(order=1,periods=[],periods_dim=1n_units=[32,32,32],learning_rate=0.1,regL2=0.001,epochs=16,train_every=32,std_targets=0.1,std_weights=1.0)"
non_contextual_default_rnn = "ThompsonRecurrentNetwork(n_units=[32,32,32],learning_rate=0.01,regL2=0.001,epochs=16,train_every=32,std_targets=0.1,std_weights=0.5)"

given_best_ffnn_policy = ''
given_best_rnn_policy = ''

def get_empirical_regret(frame, bandit, trial_length):
    ecr = bandit.expected_cumulative_rewards(trial_length)
    return ecr - frame['Return']

def main():

    # Policies to be considered in the plot, more will be appended later.
    list_policies = ['Random()']

    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Experiments directory.')
    parser.add_argument('--computebest', help='Find the best (R)NN conifguration',
                        action='store_true')
    parser.add_argument('--nondefaultasbest', help='Assumes the config other than default is the best',
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

    reference = configs[dirs[0]]

    # Consistency check
    bandit_settings = ['bandit', 'bandit_parameters','trial_length']

    for d, config in configs.items():
        for p in bandit_settings:
            if config[p] != reference[p]:
                if p == "trial_length":
                    print("\nThe trial length is different\n")
                else:
                    print("\nThe difference is in ", p)
                    print("Current config: \n")
                    print(config[p])
                    print("Reference config: \n")
                    print(reference[p])


    if reference['bandit'] in non_contextual_bandits.keys():
        default_ffnn_policy = non_contextual_default_ffnn
        default_rnn_policy = non_contextual_default_rnn

    elif reference['bandit'] in contextual_bandits.keys():
        default_ffnn_policy = contextual_default_ffnn
        default_rnn_policy = contextual_default_rnn

    else:
        default_ffnn_policy = contextual_default_ffnn
        default_rnn_policy = contextual_default_rnn

    default_rnn_policy = default_rnn_policy.replace(' ', '')
    default_ffnn_policy = default_ffnn_policy.replace(' ','')

    list_policies.append(default_ffnn_policy)
    list_policies.append(default_rnn_policy)

    df = pd.DataFrame()
    for d in dirs:
        if os.path.exists(os.path.join(d, 'trial.csv')):
            frame = pd.read_csv(os.path.join(d, 'trial.csv'))
            frame.Policy = frame.Policy.str.replace(' ', '')
            frame = frame.sort_values(by='Pull')

            # Currently, we report the empirical regret for the non-contextual and contextual bandit problems.
            # This will be made consistent in the next version of the paper.
            if reference['bandit'] in linear_bandits.keys():
                frame['Regret'] = frame['Regret']
            elif reference['bandit'] in non_contextual_bandits.keys():
                bandit = non_contextual_bandits[reference['bandit']](**reference['bandit_parameters'])
                frame['Regret'] = get_empirical_regret(frame, bandit, reference['trial_length'])
            elif reference['bandit'] in contextual_bandits.keys():
                bandit = contextual_bandits[reference['bandit']](**reference['bandit_parameters'])
                frame['Regret'] = get_empirical_regret(frame, bandit, reference['trial_length'])

            df = df.append(frame)
        else:
            msg = 'Warning: missing trial {0} for {1}({2}).'
            print(msg.format(d, configs[d]['policy'],
                             configs[d]['policy_parameters']))

    last = df[df['Pull'] == reference['trial_length']]
    last = last.loc[:, ['Return', 'Policy']]

    p_group = last.groupby('Policy').mean().sort_values(by='Return',
                                                     ascending=False)

    with pd.option_context('display.max_colwidth', -1):
        print(p_group)

    # Add the best conventional bandit algorithms to the list of policies if they
    # are present.
    if reference['bandit'] in linear_bandits.keys():
        if p_group.index.str.contains('D_LinUCB').any():
            D_LinUCB_results = p_group[p_group.index.str.contains('D_LinUCB')]
            best_D_LinUCB_policy = D_LinUCB_results.Return.idxmax()
            list_policies.append(best_D_LinUCB_policy)

        if p_group.index.str.contains('SW_LinUCB').any():
            SW_LinUCB_results = p_group[p_group.index.str.contains('SW_LinUCB')]
            best_SW_LinUCB_policy = SW_LinUCB_results.Return.idxmax()
            list_policies.append(best_SW_LinUCB_policy)


    elif reference['bandit'] in non_contextual_bandits.keys():
        if p_group.index.str.contains('D_UCB').any():
            D_UCB_results = p_group[p_group.index.str.contains('D_UCB')]
            best_D_UCB_policy = D_UCB_results.Return.idxmax()
            list_policies.append(best_D_UCB_policy)

        if p_group.index.str.contains('SW_UCB').any():
            SW_UCB_results = p_group[p_group.index.str.contains('SW_UCB')]
            best_SW_UCB_policy = SW_UCB_results.Return.idxmax()
            list_policies.append(best_SW_UCB_policy)


    # Add the best neural policies

    ffnn_policies = p_group[p_group.index.str.contains('FeedforwardNetwork')]
    rnn_policies = p_group[p_group.index.str.contains('RecurrentNetwork')]

    if args.computebest:
        best_ffnn_policy = ffnn_policies.Return.idxmax()
        best_rnn_policy = rnn_policies.Return.idxmax()

    elif args.nondefaultasbest:
        # Works when we have run the experiment only with default and optionally
        # another policy that was the best during the hyperparameter search.
        # Eg. if there is only one RNN default policy, then we assume there is
        # no separate best RNN policy

        if ((len(ffnn_policies) > 2) or (len(rnn_policies) > 2)):
            raise Exception('More than 2 (R)NN policies. Ambigous which non-default policy is best.')

        best_ffnn_policy = ffnn_policies.index[ffnn_policies.index != default_ffnn_policy]
        best_rnn_policy = rnn_policies.index[rnn_policies.index != default_rnn_policy]

        if len(best_ffnn_policy) > 0:
            best_ffnn_policy = best_ffnn_policy.values[0]
        else:
            best_ffnn_policy = default_ffnn_policy

        if len(best_rnn_policy) > 0:
            best_rnn_policy = best_rnn_policy.values[0]
        else:
            best_rnn_policy = default_rnn_policy

    else:
        # Assumes best_ffnn_policy and best_rnn_policy and provided along with
        # the default configurations

        best_ffnn_policy = given_best_ffnn_policy
        best_rnn_policy = given_best_rnn_policy


    list_policies.append(best_ffnn_policy)
    list_policies.append(best_rnn_policy)


    # Set the colour palette - keep consistent colours


    c_list = sns.color_palette()
    c_palette = {'Random' : c_list[7],
     'Best RNN': c_list[0],
     'Best NN': c_list[1],
     'Default RNN': c_list[2],
     'Default NN': c_list[3],
     'SW-UCB':c_list[4],
     'D-UCB': c_list[5]}

    if reference['bandit'] in linear_bandits.keys():
        c_palette = {'Random' : c_list[7],
          'Best RNN': c_list[0],
          'Best NN': c_list[1],
          'Default RNN': c_list[2],
          'Default NN': c_list[3],
          'SW-LinUCB':c_list[4],
          'D-LinUCB': c_list[5]}

    plot_df = df[df['Policy'].isin(list_policies)]

    # Will store the name to be used in the legend
    plot_df['Policy_newnames'] = ''

    plot_df.loc[plot_df.Policy == best_rnn_policy, 'Policy_newnames'] = 'Best RNN'
    plot_df.loc[plot_df.Policy == best_ffnn_policy, 'Policy_newnames'] = 'Best NN'

    plot_df.loc[plot_df.Policy.str.contains('Random'), 'Policy_newnames'] = 'Random'

    plot_df.loc[plot_df.Policy.str.contains('D_UCB'), 'Policy_newnames'] = 'D-UCB'
    plot_df.loc[plot_df.Policy.str.contains('SW_UCB'), 'Policy_newnames'] = 'SW-UCB'

    plot_df.loc[plot_df.Policy.str.contains('D_LinUCB'), 'Policy_newnames'] = 'D-LinUCB'
    plot_df.loc[plot_df.Policy.str.contains('SW_LinUCB'), 'Policy_newnames'] = 'SW-LinUCB'

    plot_df.loc[plot_df.Policy == default_rnn_policy, 'Policy_newnames'] = 'Default RNN'
    plot_df.loc[plot_df.Policy == default_ffnn_policy, 'Policy_newnames'] = 'Default NN'

    plot_df = plot_df.sort_values(by='Policy_newnames')
    del plot_df['Policy']
    plot_df.rename(columns = {'Policy_newnames':'Policy'}, inplace=True)

    # Plot the regret

    plt.figure(figsize=(16,9))

    plot_df = plot_df[plot_df.Pull < int(reference['trial_length'])]

    # add ci = "sd" for faster standard deviation confidence bounds, imstead of
    # a bootstrapped estimate

    ax = sns.lineplot(x='Pull', y='Regret', hue='Policy', palette = c_palette,
                      data=plot_df, linewidth=3.0)


    plt.xlim(1, int(reference['trial_length']))
    plt.xticks(range(0, int(reference['trial_length']) + 2, 1024))
    plt.xlabel('time step')
    plt.ylabel('regret')

    # Display legend in this particular order
    handles, labels = plt.gca().get_legend_handles_labels()
    order = list(range(1, len(handles)))
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    plt.savefig('regret.pdf',  bbox_inches='tight',pad_inches = 0)

if __name__ == "__main__":
    main()
