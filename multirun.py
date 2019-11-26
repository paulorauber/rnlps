""" Runs multiple jobs in parallel in a tmux session. """

import os
import time
import datetime
import argparse


def main():
    if 'TMUX' not in os.environ:
        raise Exception('This script should be called from a tmux session.')

    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('jobs', type=int)
    args = parser.parse_args()

    dirs = [os.path.join(args.directory, d)
            for d in os.listdir(args.directory) if not d.startswith('_')]
    dirs = [d for d in dirs if os.path.isdir(d)]

    cmds = ['python3 run.py {0}'.format(d) for d in dirs]

    tokens = []
    for j in range(args.jobs):
        tokens.append(os.path.join(args.directory, '.token_{0}'.format(j)))
        os.system('touch {0}'.format(tokens[-1]))

    start_time = time.time()
    for j, cmd in enumerate(cmds, 1):
        free_tokens = []
        while len(free_tokens) < 1:
            free_tokens = list(filter(os.path.exists, tokens))
            time.sleep(1)

        os.remove(free_tokens[0])

        os.system('tmux new-window -d -t {0}'.format(j))

        os.system('tmux send-keys -t {0} \"{1}\" C-m'.format(j, cmd))

        touch = 'touch {0}'.format(free_tokens[0])
        os.system('tmux send-keys -t {0} \"{1}\" C-m'.format(j, touch))

        os.system('tmux send-keys -t {0} \"exit\" C-m'.format(j))

        print('Running command {0}/{1}.'.format(j, len(cmds)))

    free_tokens = []
    while len(free_tokens) < args.jobs:
        free_tokens = list(filter(os.path.exists, tokens))
        time.sleep(1)

    for token in tokens:
        os.remove(token)

    elapsed = datetime.timedelta(seconds=time.time() - start_time)
    print('Success. Time elapsed: {0}.'.format(elapsed))


if __name__ == "__main__":
    main()
