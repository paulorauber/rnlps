"""
    Contextual bandit environments to evaluate performance.

"""

import numpy as np
import os

class StationaryContextualBandit:
    def __init__(self, dataset, seed, err_sigma = 0.05):

        # Can also be used for real-world non-stationary problems
        # as it doesn't shuffle the data.

        self.random_state = np.random.RandomState(seed)

        if os.path.isdir("datasets/" + dataset):
            self.X = np.load("datasets/" + dataset + "/X.npy")
            self.targets = np.load("datasets/" + dataset + "/y.npy")
        else :
            raise Exception("Dataset does not exist. Check the path.")

        self.n_arms = len(np.unique(self.targets))
        self.step = 0
        self.context_dims = np.shape(self.X)[-1]
        self.err_sigma = err_sigma

    def reset(self):
        self.step = 0
        return self.X[self.step]

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        reward = 0.0
        regret = 1.0

        if arm == self.targets[self.step]:
            reward = 1.0
            regret = 0.0

        assert (reward + regret) == 1

        self.step += 1
        context = self.X[self.step]

        reward = reward + self.random_state.normal(0, self.err_sigma)
        return reward, context, regret

    def best_arms(self):
        return [self.targets[self.step]]

    def expected_cumulative_rewards(self, trial_length):
        return np.cumsum(np.ones(trial_length))


    def __repr__(self):
        r = 'StationaryContextualBandit(n_arms={0}, X_dims={1})'
        return r.format(self.n_arms, np.shape(self.X))


class FlippingContextualBandit:

    def __init__(self, dataset, half_period, seed, err_sigma = 0.05):

        self.random_state = np.random.RandomState(seed)


        if os.path.isdir("datasets/" + dataset):
            self.X = np.load("datasets/" + dataset + "/X.npy")
            self.targets = np.load("datasets/" + dataset + "/y.npy")
        else :
            raise Exception("Dataset does not exist. Check the path.")

        self.n_arms = len(np.unique(self.targets))
        self.step = 0
        self.half_period = half_period
        self.context_dims = np.shape(self.X)[-1]

        self.flipped = 0
        self.err_sigma = err_sigma

    def reset(self):
        self.step = 0
        return self.X[self.step]

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        if (arm == self.targets[self.step]) & (self.flipped == 0):
            reward = 1.0
            regret = 0.0
        elif (arm == ((self.n_arms - 1 - self.targets[self.step]) % self.n_arms) & self.flipped == 1):
            reward = 1.0
            regret = 0.0
        else:
            reward = 0.0
            regret = 1.0

        assert (reward + regret) == 1

        self.step += 1
        context = self.X[self.step]

        if self.step % self.half_period == 0:
            self.flipped = (self.flipped + 1) % 2

        reward = reward + self.random_state.normal(0, self.err_sigma)

        return reward, context, regret

    def best_arms(self):

        best = self.targets[self.step]
        if self.flipped:
            best = (self.n_arms - 1 - self.targets[self.step]) % self.n_arms

        return [best]


    def expected_cumulative_rewards(self, trial_length):
        return np.cumsum(np.ones(trial_length))


    def __repr__(self):
        r = 'FlippingContextualBandit(n_arms={0}, X_dims={1}, half_period={2})'
        return r.format(self.n_arms, np.shape(self.X), self.half_period)

contextual_bandits = {'StationaryContextualBandit': StationaryContextualBandit,
            'FlippingContextualBandit': FlippingContextualBandit}
