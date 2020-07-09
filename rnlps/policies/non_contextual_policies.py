"""
    Policies for non-contextual bandit problems.

"""

import numpy as np
import tensorflow as tf
import contextlib
from termcolor import cprint

from rnlps.policies.base import Trial, Policy
from rnlps.policies.base import BaseOracle, BaseFixed, BaseRandom
from rnlps.policies.base import BaseThompsonRecurrentNetwork, BaseThompsonSinFeedforwardNetwork

@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

class Oracle(BaseOracle):
    pass


class Fixed(BaseFixed):
    pass


class Random(BaseRandom):
    pass


class UCB(Policy):
    def __init__(self, bandit):
        Policy.__init__(self, bandit)

    def select(self, trial):
        for arm in range(trial.n_arms):
            if trial.pulls[arm] < 1:
                return arm

        ft = 1 + (trial.length + 1)*(np.log(trial.length + 1)**2)
        bonus = np.sqrt((2.*np.log(ft))/trial.pulls)

        indices = trial.average_rewards() + bonus

        return np.argmax(indices)

    def __repr__(self):
        return 'UCB()'


class SW_UCB(Policy):
    """ Sliding-window UCB implementation based on the paper,
    'On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems' -
    https://arxiv.org/pdf/0805.3415.pdf """

    def __init__(self, bandit, tau, seed, ksi = 0.6):
        Policy.__init__(self, bandit)
        self.tau = tau
        self.ksi = ksi
        self.random_state = np.random.RandomState(seed)

    def select(self, trial):
        for arm in self.random_state.permutation(trial.n_arms):
            if trial.pulls[arm] < 1:
                return arm

        # Consider only the last tau steps

        arms_tau = np.array(trial.arms[-self.tau:])
        rewards_tau = np.array(trial.rewards[-self.tau:])

        returns_tau = np.zeros(trial.n_arms)
        N_tau = np.zeros(trial.n_arms)

        for arm in range(trial.n_arms):
            N_tau[arm] = np.sum(arms_tau == arm)
            returns_tau[arm] = np.sum((arms_tau == arm) * rewards_tau)

        out = np.full(trial.n_arms, float('inf'))
        where = (N_tau > 0)

        avg_rewards_tau = np.divide(returns_tau, N_tau, out=out, where=where)

        out2 = np.full(trial.n_arms, float('inf'))
        where2 = (N_tau > 0)

        ins_sqrt_term = np.divide(self.ksi * np.log(min(self.tau, trial.length)), N_tau, out=out2, where=where2)
        bonus = np.sqrt(ins_sqrt_term)

        indices = avg_rewards_tau + bonus

        # break ties in a random way

        max_arms = np.argwhere(indices == np.amax(indices))
        chosen_arm = self.random_state.choice(max_arms.flatten())


        return chosen_arm

    def __repr__(self):
        return 'SW_UCB(tau={0}, ksi={1})'.format(self.tau, self.ksi)

class D_UCB(Policy):
    """ Discounted UCB implementation based on the paper,
    'On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems' -
    https://arxiv.org/pdf/0805.3415.pdf """

    def __init__(self, bandit, gamma, seed, ksi = 0.6):
        Policy.__init__(self, bandit)
        self.gamma = gamma
        self.ksi = ksi
        self.random_state = np.random.RandomState(seed)

    def select(self, trial):
        for arm in self.random_state.permutation(trial.n_arms):
            if trial.pulls[arm] < 1:
                return arm

        arms_all = np.array(trial.arms)
        rewards_all = np.array(trial.rewards)

        discount_arr = np.flip(np.cumprod(self.gamma * np.ones(trial.length - 1)))
        discount_arr = np.append(discount_arr, 1.0)

        discounted_rewards = rewards_all * discount_arr

        returns_gamma = np.zeros(trial.n_arms)
        N_gamma = np.zeros(trial.n_arms)

        for arm in range(trial.n_arms):
            N_gamma[arm] = np.sum((arms_all == arm) * discount_arr)
            returns_gamma[arm] = np.sum((arms_all == arm) * discounted_rewards)

        out = np.full(trial.n_arms, float('inf'))
        where = (N_gamma > 0)


        avg_rewards_gamma = np.divide(returns_gamma, N_gamma, out=out, where=where)

        n_tot_gamma = np.sum(N_gamma)

        out2 = np.full(trial.n_arms, float('inf'))
        where2 = (N_gamma > 0)

        ins_sqrt_term = np.divide(self.ksi * np.log(n_tot_gamma), N_gamma, out=out2, where=where2)
        bonus = 2 * np.sqrt(ins_sqrt_term)
        indices = avg_rewards_gamma + bonus

        # break ties in a random way

        max_arms = np.argwhere(indices == np.amax(indices))
        chosen_arm = self.random_state.choice(max_arms.flatten())

        return chosen_arm

    def __repr__(self):
        return 'D_UCB(gamma={0}, ksi={1})'.format(self.gamma, self.ksi)


class ThompsonSamplingBernoulli(Policy):
    """ Thompson sampling for the k-armed bernoulli bandit. """

    def __init__(self, bandit, a, b, seed):
        Policy.__init__(self, bandit)

        self.a = a
        self.b = b

        self.random_state = np.random.RandomState(seed=seed)

    def select(self, trial):
        a = trial.returns
        b = trial.pulls - trial.returns

        means = np.zeros(trial.n_arms)
        for i in range(trial.n_arms):
            means[i] = self.random_state.beta(a[i] + self.a, b[i] + self.b)

        return np.argmax(means)

    def __repr__(self):
        return 'ThompsonSamplingBernoulli(a={0}, b={1})'.format(self.a, self.b)


class ThompsonRecurrentNetwork(BaseThompsonRecurrentNetwork):
    """ Recurrent neural-linear: Thompson sampling based policy by using
        Bayesian linear regression on the representation(context) generated by
        the penultimate layer of the recurrent architecture. """

    def __init__(self, bandit, n_units, learning_rate, regularise_lambda,
                 epochs, train_every, std_targets, std_weights, verbose, seed):

        BaseThompsonRecurrentNetwork.__init__(self, bandit, n_units,
        learning_rate, regularise_lambda, epochs, train_every, std_targets,
        std_weights, verbose, seed)


    def _setup_input(self):
        """ Returns the input and input size for reward prediction by the RNN. """

        self._arms = tf.placeholder(tf.int32, shape=None, name='arms')

        rewards = tf.reshape(self._rewards[:-1], (-1, 1))
        arms_oh = tf.one_hot(self._arms, depth=self.bandit.n_arms,
                                dtype=tf.float32, name='arms_oh')

        inputs = tf.concat([rewards, arms_oh], axis=1, name='inputs')
        isize = self.bandit.n_arms + 1

        return inputs, isize


    def _get_pred_from_sampled_model(self, trial, w, state):
        """ Obtains the predicted reward for every action under the sampled
        model(w). """

        pred = np.zeros(trial.n_arms)
        for arm in range(trial.n_arms):
            feed = {self._istate: state, self._arms: [arm],
                    self._rewards: [trial.rewards[-1], 0]}

            arm_features = self.session.run(self._h_output, feed)[0]
            pred[arm] = arm_features.dot(w)

        return pred

    def _get_feed_for_rnn(self, trial):
        """ Returns the input feed for the action selection by the policy. """
        feed = {self._arms: trial.arms, self._rewards: [0.] + trial.rewards}
        return feed



class ThompsonSinFeedforwardNetwork(BaseThompsonSinFeedforwardNetwork):
    """ Neural-linear: Thompson sampling with Bayesian linear regression on
        the representation generated by a FFNN fed with handcrafted context and
         sinusoidal input units."""

    def __init__(self, bandit, order, periods, periods_dim, n_units,
                 learning_rate, regularise_lambda, epochs, train_every,
                 std_targets, std_weights, verbose, seed):

        BaseThompsonSinFeedforwardNetwork.__init__(self, bandit, order, periods, periods_dim, n_units,
                     learning_rate, regularise_lambda, epochs, train_every,
                     std_targets, std_weights, verbose, seed)

    def _setup_input_size(self):
        """ Returns the number of units in the input layer, before and after
        including the sinusoidal units. """

        isize = len(self.periods)*2
        isize += self.bandit.n_arms + (self.bandit.n_arms + 1) * self.order

        #add more units for the periodic transfromation

        isize_pre = isize + 1
        isize_post = isize + self.periods_dim

        return isize_pre, isize_post

    def _example(self, arms, rewards, contexts, t):
        """ Creates one sample of the handcrafted context-target pair. Used to
        generate the training data for training the NN representation model. """

        # Dataset to predict the reward based on handcrafted context and current action.

        if t < self.order:
            raise Exception('Incomplete observation.')

        arms_oh = np.eye(self.bandit.n_arms)

        observation = []

        # Can be used for hard-coded periods
        for p in self.periods:
            angle = (2*np.pi*t)/p
            observation += [np.cos(angle), np.sin(angle)]

        observation.extend(arms_oh[arms[t]])

        for i in range(self.order):
            observation.extend(arms_oh[arms[t - i - 1]])
            observation.append(rewards[t - i - 1])

        observation.append(t) #add the raw time-step as the new input
        target = rewards[t]

        return observation, target

non_contextual_policies = {'Oracle': Oracle,
            'Fixed': Fixed,
            'Random': Random,
            'UCB': UCB,
            'SW_UCB' : SW_UCB,
            'D_UCB' : D_UCB,
            'ThompsonSamplingBernoulli': ThompsonSamplingBernoulli,
            'ThompsonRecurrentNetwork': ThompsonRecurrentNetwork,
            'ThompsonSinFeedforwardNetwork': ThompsonSinFeedforwardNetwork}
