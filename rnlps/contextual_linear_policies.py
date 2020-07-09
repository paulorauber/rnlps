"""
    Policies for linear bandit problems.

"""

import numpy as np
import tensorflow as tf
import contextlib
from termcolor import cprint
from math import log

from rnlps.base import Trial, Policy
from rnlps.base import BaseOracle, BaseFixed, BaseRandom
from rnlps.base import BaseThompsonRecurrentNetwork, BaseThompsonSinFeedforwardNetwork


@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class ContextualLinearTrial(Trial):
    def __init__(self, n_arms):
        self.arm_embeddings = [] # actions are no longer one hot
        Trial.__init__(self, n_arms)

    def update_contexts(self, context):
        # Don't append here. No need to store the entire set of previously available arms
        self.contexts = context

    def append(self, arm, reward, context, regret):
        self.arm_embeddings.append(self.contexts[arm])
        Trial.append(self, arm, reward, context, regret)

class ContextualLinearPolicy(Policy):
    def __init__(self, bandit):
        Policy.__init__(self, bandit)
        self.contextual_bandit = 0

        # we use the context to only collect the available arms
        # prediction depends on the arm chosen

    def interact(self, trial_length):
        trial = ContextualLinearTrial(self.bandit.n_arms)

        reset_context = self.bandit.reset()
        trial.update_contexts(reset_context)

        for i in range(trial_length):
            arm = self.select(trial)
            reward, context, regret = self.bandit.pull(arm)

            trial.append(arm, reward, context, regret)

        return trial


class Oracle(ContextualLinearPolicy, BaseOracle):
    """Selects the best available action at every time step. """

    def __init__(self, bandit):
        ContextualLinearPolicy.__init__(self, bandit)


class Fixed(ContextualLinearPolicy, BaseFixed):
    """Selects a fixed chosen action at every time step. """

    def __init__(self, bandit, arm):
        ContextualLinearPolicy.__init__(self, bandit)
        self.arm = arm

class Random(ContextualLinearPolicy, BaseRandom):
    """Selects a random action at every time step. """

    def __init__(self, bandit, seed):
        ContextualLinearPolicy.__init__(self, bandit)
        self.random_state = np.random.RandomState(seed)

class LinUCB(ContextualLinearPolicy):
    def __init__(self, bandit, delta, alpha, lambda_reg, sigma_noise, seed):
        ContextualLinearPolicy.__init__(self, bandit)
        self.random_state = np.random.RandomState(seed)

        self.arm_dims = bandit.dimension
        self.delta = delta
        self.alpha = alpha # multiplicative constant to tune exploration bonus
        self.sigma_noise = sigma_noise

        self.ucbs = np.zeros(bandit.n_arms)
        self.theta_hat = np.zeros(self.arm_dims)
        self.lambda_reg = lambda_reg

        self.V = self.lambda_reg * np.eye(self.arm_dims)
        self.inv_V = (1/self.lambda_reg) * np.eye(self.arm_dims)
        self.b = np.zeros(self.arm_dims)

        # assumes that norm of theta_star and actions are bounded by 1.

    def select(self, trial):

        # Update variables based on previous arm and reward

        if trial.length >= 1:
            self.V = self.V + np.outer(trial.arm_embeddings[-1], trial.arm_embeddings[-1])
            self.b = self.b + trial.arm_embeddings[-1] * trial.rewards[-1]

            self.inv_V = np.linalg.pinv(self.V)
            self.theta_hat = np.dot(self.inv_V, self.b)

        sqrt_beta_t = np.sqrt(self.lambda_reg)
        sqrt_beta_t += self.sigma_noise * np.sqrt(2*log(1/self.delta) + self.arm_dims * log(1 + trial.length/(self.lambda_reg * self.arm_dims)))

        for arm in range(trial.n_arms):
            invV_arm = np.dot(self.inv_V, trial.contexts[arm])
            norm_a_with_invV = np.sqrt(np.dot(trial.contexts[arm], invV_arm))
            self.ucbs[arm] = np.dot(self.theta_hat, trial.contexts[arm]) + self.alpha * sqrt_beta_t * norm_a_with_invV

        max_arms = np.argwhere(self.ucbs == np.amax(self.ucbs))
        chosen_arm = self.random_state.choice(max_arms.flatten())

        return chosen_arm


    def __repr__(self):
        r = 'LinUCB(delta={0}, alpha={1}, lambda_reg={2}, sigma_noise={3})'
        return r.format(self.delta, self.alpha, self.lambda_reg, self.sigma_noise)

class SW_LinUCB(ContextualLinearPolicy):
    """ Implementation of SW-LinUCB, by by Cheung et. al -
        'Learning to optimize under Non-Stationarity', AISTATS19.
         Based on the code available at
        https://github.com/YRussac/WeightedLinearBandits/blob/master/D_LinUCB_class.py """

    def __init__(self, bandit, delta, alpha, tau, lambda_reg, sigma_noise, seed):
        ContextualLinearPolicy.__init__(self, bandit)

        self.random_state = np.random.RandomState(seed)

        self.arm_dims = bandit.dimension
        self.delta = delta
        self.alpha = alpha # multiplicative constant to tune exploration bonus
        self.tau = tau # window size
        self.sigma_noise = sigma_noise

        self.ucbs = np.zeros(bandit.n_arms)
        self.theta_hat = np.zeros(self.arm_dims)
        self.lambda_reg = lambda_reg

        self.V = self.lambda_reg * np.eye(self.arm_dims)
        self.inv_V = (1/self.lambda_reg) * np.eye(self.arm_dims)
        self.b = np.zeros(self.arm_dims)

        self.beta = np.sqrt(self.lambda_reg) + self.sigma_noise * np.sqrt(self.arm_dims * np.log((1 + self.tau/self.lambda_reg)/self.delta))
        # beta_t is fixed in this algorithm

    def select(self, trial):

        # Update variables based on previous arm and reward

        if trial.length >= 1:

            if trial.length <= self.tau:
                self.V = self.V + np.outer(trial.arm_embeddings[-1], trial.arm_embeddings[-1])
                self.b = self.b + trial.arm_embeddings[-1] * trial.rewards[-1]

                self.inv_V = np.linalg.pinv(self.V)
                self.theta_hat = np.dot(self.inv_V, self.b)

            else:

                a_removal = trial.arm_embeddings[-(self.tau + 1)]
                r_removal = trial.rewards[-(self.tau + 1)]

                self.V = self.V + np.outer(trial.arm_embeddings[-1], trial.arm_embeddings[-1]) - np.outer(a_removal, a_removal)
                self.b = self.b + trial.arm_embeddings[-1] * trial.rewards[-1] - a_removal * r_removal

                self.inv_V = np.linalg.pinv(self.V)
                self.theta_hat = np.dot(self.inv_V, self.b)

        for arm in range(trial.n_arms):
            invV_arm = np.dot(self.inv_V, trial.contexts[arm])
            norm_a_with_invV = np.sqrt(np.dot(trial.contexts[arm], invV_arm))
            self.ucbs[arm] = np.dot(self.theta_hat, trial.contexts[arm]) + self.alpha * self.beta * norm_a_with_invV


        max_arms = np.argwhere(self.ucbs == np.amax(self.ucbs))
        chosen_arm = self.random_state.choice(max_arms.flatten())

        return chosen_arm


    def __repr__(self):
        r = 'SW_LinUCB(delta={0}, alpha={1}, tau={2}, lambda_reg={3}, sigma_noise={4})'
        return r.format(self.delta, self.alpha, self.tau, self.lambda_reg, self.sigma_noise)


class D_LinUCB(ContextualLinearPolicy):
    """ Implementation of D-LinUCB, by Russac et. al -
        'Weighted Linear Bandits in Non-Stationary Environments', NeurIPS19.
         Based on the code available at
        https://github.com/YRussac/WeightedLinearBandits/blob/master/D_LinUCB_class.py """


    def __init__(self, bandit, delta, alpha, gamma, lambda_reg, sigma_noise, seed):
        ContextualLinearPolicy.__init__(self, bandit)
        self.random_state = np.random.RandomState(seed)

        self.arm_dims = bandit.dimension
        self.delta = delta
        self.alpha = alpha # multiplicative constant to tune exploration bonus
        self.gamma = gamma # discount factor
        self.sigma_noise = sigma_noise

        self.ucbs = np.zeros(bandit.n_arms)
        self.theta_hat = np.zeros(self.arm_dims)
        self.lambda_reg = lambda_reg

        self.V = self.lambda_reg * np.eye(self.arm_dims)
        self.V_tilde = self.lambda_reg * np.eye(self.arm_dims)
        self.inv_V = (1/self.lambda_reg) * np.eye(self.arm_dims)
        self.b = np.zeros(self.arm_dims)

        self.gamma2_t = 1


    def select(self, trial):

        # Update variables based on previous arm and reward

        if trial.length >= 1:

            self.gamma2_t *= self.gamma ** 2

            self.V = self.gamma * self.V + np.outer(trial.arm_embeddings[-1], trial.arm_embeddings[-1]) + (1 - self.gamma) * self.lambda_reg * np.eye(self.arm_dims)

            #Their update - according to the code
            self.V_tilde = (self.gamma**2) * self.V + np.outer(trial.arm_embeddings[-1], trial.arm_embeddings[-1]) + (1 - self.gamma**2) * self.lambda_reg * np.eye(self.arm_dims)

            # The update according to the paper?
            #self.V_tilde = (self.gamma**2) * self.V_tilde + np.outer(trial.arm_embeddings[-1], trial.arm_embeddings[-1]) + (1 - self.gamma**2) * self.lambda_reg * np.eye(self.arm_dims)

            self.b = self.gamma * self.b + trial.arm_embeddings[-1] * trial.rewards[-1]

            self.inv_V = np.linalg.pinv(self.V)
            self.theta_hat = np.dot(self.inv_V, self.b)

        beta_t = np.sqrt(self.lambda_reg)
        beta_t += self.sigma_noise * np.sqrt(2*log(1/self.delta) + self.arm_dims * np.log(1 + (1-self.gamma2_t)/(self.arm_dims * self.lambda_reg * (1 - self.gamma**2))))

        for arm in range(trial.n_arms):
            invV_arm = np.dot(np.matmul(self.inv_V,np.matmul(self.V_tilde,self.inv_V)), trial.contexts[arm])
            norm_a_with_invV = np.sqrt(np.dot(trial.contexts[arm], invV_arm))
            self.ucbs[arm] = np.dot(self.theta_hat, trial.contexts[arm]) + self.alpha * beta_t * norm_a_with_invV

        max_arms = np.argwhere(self.ucbs == np.amax(self.ucbs))
        chosen_arm = self.random_state.choice(max_arms.flatten())

        return chosen_arm

    def __repr__(self):
        r = 'D_LinUCB(delta={0}, alpha={1}, gamma={2}, lambda_reg={3}, sigma_noise={4})'
        return r.format(self.delta, self.alpha, self.gamma, self.lambda_reg, self.sigma_noise)

class ThompsonRecurrentNetwork(ContextualLinearPolicy, BaseThompsonRecurrentNetwork):
    """ Recurrent neural-linear: Thompson sampling based policy by using
        Bayesian linear regression on the representation(context) generated by
        the penultimate layer of the recurrent architecture. """

    def __init__(self, bandit, n_units, learning_rate, regularise_lambda, epochs,
                 train_every, std_targets, std_weights, verbose, seed):

        # Overrides the __init__ method from BaseThompsonRecurrentNetwork

        ContextualLinearPolicy.__init__(self, bandit)

        self.n_units = n_units
        if len(self.n_units) < 2:
            raise Exception('Invalid number of layers.')

        self.learning_rate = learning_rate
        self.regularise_lambda = regularise_lambda
        self.epochs = epochs
        self.train_every = train_every

        self.var_targets = std_targets**2.
        self.var_weights = std_weights**2.
        self.one_over_lambda = self.var_weights/self.var_targets

        self.verbose = verbose

        self.random_state = np.random.RandomState(seed=seed)
        self.arm_dims = bandit.dimension

        self._setup(seed)

    def select(self, trial):
        """ Selects which arm to play in the current round. """

        # Overrides the select method in BaseThompsonRecurrentNetwork
        # we don't need to pull every arm once here

        # The first pull is just arm = 0, after that we use the network
        if trial.length < 1:
            return 0

        else:
            return self._select_from_policy(trial)

    def _setup_input(self):
        """ Returns the input and input size for reward prediction by the RNN. """

        self._arms = tf.placeholder(tf.float32, shape=[None, self.arm_dims], name='arms')

        rewards = tf.reshape(self._rewards[:-1], (-1, 1))
        # past rewards seen as features for prediction, first value is 0, last one can't be used as a feature for the next step

        arms_emb = self._arms

        inputs = tf.concat([rewards, arms_emb], axis=1, name='inputs')
        isize = self.arm_dims + 1

        return inputs, isize

    def _get_pred_from_sampled_model(self, trial, w, state):
        """ Obtains the predicted reward for every action under the sampled
        model(w). """

        pred = np.zeros(trial.n_arms)
        for arm in range(trial.n_arms):
            feed = {self._istate: state, self._arms: np.reshape(trial.contexts[arm], (1, -1)),
                    self._rewards: [trial.rewards[-1], 0]}#, self._contexts: [trial.contexts[-1]]}

            arm_features = self.session.run(self._h_output, feed)[0]
            pred[arm] = arm_features.dot(w)

        return pred

    def _get_feed_for_rnn(self, trial):
        """ Returns the input feed for the action selection by the policy. """

        feed = {self._arms: trial.arm_embeddings, self._rewards: [0.] + trial.rewards}
        return feed



class ThompsonSinFeedforwardNetwork(ContextualLinearPolicy, BaseThompsonSinFeedforwardNetwork):
    """ Neural-linear: Thompson sampling with Bayesian linear regression on
        the representation generated by a FFNN fed with handcrafted context and
         sinusoidal input units."""

    def __init__(self, bandit, order, periods, periods_dim, n_units,
                 learning_rate, regularise_lambda, epochs, train_every,
                 std_targets, std_weights, verbose, seed):

        # Overrides the __init__ method from BaseThompsonSinFeedforwardNetwork

        ContextualLinearPolicy.__init__(self, bandit)

        self.order = order
        self.periods = np.array(periods)
        self.periods_dim = periods_dim
        self.n_units = n_units

        self.learning_rate = learning_rate
        self.regularise_lambda = regularise_lambda
        self.epochs = epochs
        self.train_every = train_every

        self.var_targets = std_targets**2.
        self.var_weights = std_weights**2.
        self.one_over_lambda = self.var_weights/self.var_targets

        self.verbose = verbose
        self.arm_dims = bandit.dimension

        self.random_state = np.random.RandomState(seed=seed)

        self._setup(seed)

    def select(self, trial):
        """ Selects which arm to play in the current round. """

        # Overrides the select method in BaseThompsonSinFeedforwardNetwork
        # we don't need to pull every arm once here

        if trial.length < (self.order + 1):
            return np.argmin(trial.pulls)

        else:
            return self._select_from_policy(trial)

    def _get_observation_target_pair(self, trial):
        """ Creates the dataset for training the NN representation model. """

        # Overrides the _get_observation_target_pair method in BaseThompsonSinFeedforwardNetwork
        # uses the arm embeddings for prediction

        observations = []
        targets = []
        for t in range(self.order, trial.length):
            observation, target = self._example(trial.arm_embeddings, trial.rewards, t)

            observations.append(observation)
            targets.append(target)

        return observations, targets

    def _get_pred_from_sampled_model(self, trial, w):
        """ Obtains the predicted reward for every action under the sampled
        model(w). """

        # Overrides the _get_pred_from_sampled_model method in BaseThompsonSinFeedforwardNetwork
        # uses the arm embeddings for prediction

        pred = np.zeros(trial.n_arms)
        for arm in range(trial.n_arms):
            observation, _ = self._example(trial.arm_embeddings + [trial.contexts[arm]],
                                           trial.rewards + [0.],
                                           trial.length)
            feed = {self._observations: [observation]}
            arm_features = self.session.run(self._h_output, feed)[0]
            pred[arm] = arm_features.dot(w)

        return pred

    def _setup_input_size(self):
        """ Returns the number of units in the input layer, before and after
        including the sinusoidal units. """

        isize = len(self.periods)*2
        isize += self.arm_dims + (self.arm_dims + 1) * self.order

        #add more units for the periodic transfromation
        isize_pre = isize + 1
        isize_post = isize + self.periods_dim

        return isize_pre, isize_post


    def _example(self, arm_embeddings, rewards, t):
        """ Creates one sample of the handcrafted context-target pair. Used to
        generate the training data for training the NN representation model. """

        if t < self.order:
            raise Exception('Incomplete observation.')

        observation = []
        for p in self.periods:
            angle = (2*np.pi*t)/p
            observation += [np.cos(angle), np.sin(angle)]

        observation.extend(arm_embeddings[t])

        for i in range(self.order):
            observation.extend(arm_embeddings[t - i - 1])
            observation.append(rewards[t - i - 1])

        observation.append(t) #add the raw time-step as the new input
        target = rewards[t]

        return observation, target



contextual_linear_policies = {'Oracle': Oracle,
            'Fixed': Fixed,
            'Random': Random,
            'LinUCB': LinUCB,
            'SW_LinUCB': SW_LinUCB,
            'D_LinUCB': D_LinUCB,
            'ThompsonRecurrentNetwork': ThompsonRecurrentNetwork,
            'ThompsonSinFeedforwardNetwork': ThompsonSinFeedforwardNetwork}
