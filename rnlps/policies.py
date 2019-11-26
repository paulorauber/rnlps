""" Policies for bandit problems. """

import numpy as np
import tensorflow as tf
import contextlib
from termcolor import cprint


@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class Trial:
    def __init__(self, n_arms):
        self.n_arms = n_arms

        self.returns = np.zeros(n_arms, dtype=np.float)
        self.pulls = np.zeros(n_arms, dtype=np.int)

        self.arms = []
        self.rewards = []
        self.length = 0

    def append(self, arm, reward):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        self.arms.append(arm)
        self.rewards.append(reward)
        self.length += 1

        self.returns[arm] += reward
        self.pulls[arm] += 1

    def average_rewards(self):
        out = np.full(self.n_arms, float('inf'))
        where = (self.pulls > 0)

        return np.divide(self.returns, self.pulls, out=out, where=where)

    def cumulative_rewards(self):
        return np.cumsum(self.rewards)


class Policy:
    def __init__(self, bandit):
        self.bandit = bandit

    def select(self, trial):
        raise NotImplementedError()

    def interact(self, trial_length):
        trial = Trial(self.bandit.n_arms)

        for _ in range(trial_length):
            arm = self.select(trial)
            reward = self.bandit.pull(arm)

            trial.append(arm, reward)

        return trial


class Oracle(Policy):
    def __init__(self, bandit):
        Policy.__init__(self, bandit)

    def select(self, trial):
        return self.bandit.best_arms()[0]

    def __repr__(self):
        return 'Oracle()'


class Fixed(Policy):
    def __init__(self, bandit, arm):
        Policy.__init__(self, bandit)
        self.arm = arm

    def select(self, trial):
        return self.arm

    def __repr__(self):
        return 'Fixed(arm={0})'.format(self.arm)


class Random(Policy):
    def __init__(self, bandit, seed):
        Policy.__init__(self, bandit)
        self.random_state = np.random.RandomState(seed)

    def select(self, trial):
        return self.random_state.randint(trial.n_arms)

    def __repr__(self):
        return 'Random()'


class EpsilonGreedy(Policy):
    def __init__(self, bandit, epsilon, seed):
        Policy.__init__(self, bandit)
        self.epsilon = epsilon
        self.random_state = np.random.RandomState(seed)

    def select(self, trial):
        if self.random_state.rand() < self.epsilon:
            return self.random_state.randint(trial.n_arms)

        return np.argmax(trial.average_rewards())

    def __repr__(self):
        return 'EpsilonGreedy(epsilon={0})'.format(self.epsilon)


class Boltzmann(Policy):
    def __init__(self, bandit, tau, seed):
        Policy.__init__(self, bandit)
        self.tau = tau
        self.random_state = np.random.RandomState(seed)

    def select(self, trial):
        for arm in range(trial.n_arms):
            if trial.pulls[arm] < 1:
                return arm

        erewards = np.exp(trial.average_rewards()/self.tau)
        erewards = erewards/np.sum(erewards)

        return self.random_state.choice(trial.n_arms, p=erewards)

    def __repr__(self):
        return 'Boltzmann(tau={0})'.format(self.tau)


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


class BanditRecurrentNetwork(Policy):
    """ RNN (LSTM) base that is trained to predict rewards. Used to create the
        context on which we perform Bayesian linear regression. """

    def __init__(self, bandit, n_units, learning_rate, regularise_lambda,
                 epochs, train_every, verbose, seed):
        Policy.__init__(self, bandit)

        self.n_units = n_units
        if len(self.n_units) < 2:
            raise Exception('Invalid number of layers.')

        self.learning_rate = learning_rate
        self.regularise_lambda = regularise_lambda
        self.epochs = epochs
        self.train_every = train_every

        self.verbose = verbose

        self.random_state = np.random.RandomState(seed=seed)

        self._setup(seed)

    def _setup(self, seed):
        tf.reset_default_graph()
        tf.set_random_seed(seed)

        self._arms = tf.placeholder(tf.int32, shape=None, name='arms')
        self._rewards = tf.placeholder(tf.float32, shape=None, name='rewards')

        rewards = tf.reshape(self._rewards[:-1], (-1, 1))
        arms_oh = tf.one_hot(self._arms, depth=self.bandit.n_arms,
                                dtype=tf.float32, name='arms_oh')

        inputs = tf.concat([rewards, arms_oh], axis=1, name='inputs')
        isize = self.bandit.n_arms + 1

        W = tf.Variable(tf.truncated_normal(shape=(isize, self.n_units[0])),
                                            name='W0')
        b = tf.Variable(tf.zeros(shape=(self.n_units[0])), name='b0')

        rnn_inputs = tf.matmul(inputs, W) + b
        rnn_inputs = tf.expand_dims(rnn_inputs, axis=0)

        cell = tf.contrib.rnn.LSTMCell(num_units=self.n_units[1])
        self._istate = cell.zero_state(1, dtype=tf.float32)

        rnn_outputs, \
            self._final_state = tf.nn.dynamic_rnn(cell, rnn_inputs,
                                                  initial_state=self._istate)

        self._h_output = tf.reshape(rnn_outputs, (-1, self.n_units[1]))
        for i in range(2, len(self.n_units)):
            W = tf.Variable(tf.truncated_normal(shape=(self.n_units[i - 1],
                                                       self.n_units[i])),
                            name='W{0}'.format(i))
            b = tf.Variable(tf.zeros(shape=(self.n_units[i])),
                            name='b{0}'.format(i))

            self._h_output = tf.tanh(tf.matmul(self._h_output, W) + b)

        W = tf.Variable(tf.truncated_normal(shape=(self.n_units[-1], 1)),
                        name='Wout')

        # Note: No bias in the output layer
        self._pred = tf.matmul(self._h_output, W)
        self._pred = tf.reshape(self._pred, (-1,))

        # L2 regularization on weights
        self._reg_loss = sum(tf.nn.l2_loss(tf_var)
                for tf_var in tf.trainable_variables()
                if not ("b" in tf_var.name))

        self._loss = tf.reduce_mean((self._pred - self._rewards[1:])**2) + \
                     self.regularise_lambda * self._reg_loss

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self._train = optimizer.minimize(self._loss)

        config = tf.ConfigProto(device_count={'GPU': 0})
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

    def select(self, trial):
        raise NotImplementedError()


class BoltzmannRecurrentNetwork(BanditRecurrentNetwork):
    """ Boltzmann selection policy using the RNN to predict rewards. """
    def __init__(self, bandit, n_units, learning_rate, regularise_lambda,
                 epochs, train_every, tau, verbose, seed):
        BanditRecurrentNetwork.__init__(self, bandit, n_units, learning_rate,
                                        regularise_lambda, epochs, train_every,
                                        verbose, seed)
        self.tau = tau

    def select(self, trial):
        for arm in range(trial.n_arms):
            if trial.pulls[arm] < 1:
                return arm

        feed = {self._arms: trial.arms, self._rewards: [0.] + trial.rewards}

        if trial.length % self.train_every == 0:
            for _ in range(self.epochs):
                self.session.run(self._train, feed)

        loss, state = self.session.run([self._loss, self._final_state], feed)

        pred = np.zeros(trial.n_arms)
        for arm in range(trial.n_arms):
            feed = {self._istate: state, self._arms: [arm],
                    self._rewards: [trial.rewards[-1], 0]}
            pred[arm] = self.session.run(self._pred, feed)[0]

        p = np.exp(pred/self.tau)
        p = p/np.sum(p)

        if self.verbose:
            msg = 'Pull: {0:4d}. Loss: {1:.4f}. Prediction: {2}. Policy: {3}.'

            with _printoptions(precision=4, suppress=True):
                if np.argmax(p) in self.bandit.best_arms():
                    cprint(msg.format(trial.length + 1, loss, pred, p), 'green')
                else:
                    print(msg.format(trial.length + 1, loss, pred, p))

        return self.random_state.choice(trial.n_arms, p=p)

    def __repr__(self):
        r = 'BoltzmannRecurrentNetwork(n_units={0}, learning_rate={1}, '
        r += 'epochs={2}, train_every={3}, tau={4})'
        return r.format(self.n_units, self.learning_rate, self.epochs,
                        self.train_every, self.tau)


class ThompsonRecurrentNetwork(BanditRecurrentNetwork):
    """ Recurrent neural-linear: Thompson sampling based policy by using
        Bayesian linear regression on the representation (context) generated by
        the penultimate layer of BanditRecurrentNetwork. """

    def __init__(self, bandit, n_units, learning_rate, regularise_lambda,
                 epochs, train_every, std_targets, std_weights, verbose, seed):
        BanditRecurrentNetwork.__init__(self, bandit, n_units, learning_rate,
                                        regularise_lambda, epochs, train_every,
                                        verbose, seed)
        # lambda = var_targets/var_weights
        self.var_targets = std_targets**2.
        self.var_weights = std_weights**2.

    def select(self, trial):
        """ Selects which arm to play in the current round according to Thompson sampling. """
        for arm in range(trial.n_arms):
            if trial.pulls[arm] < 1:
                return arm

        feed = {self._arms: trial.arms, self._rewards: [0.] + trial.rewards}

        if trial.length % self.train_every == 0:
            # Train the RNN weights when this condition is true
            for _ in range(self.epochs):
                self.session.run(self._train, feed)

        loss, state, observations = self.session.run([self._loss,
                                                      self._final_state,
                                                      self._h_output], feed)

        targets = np.array(trial.rewards)

        # Update the posterior for Bayesian linear regression
        one_over_lambda = self.var_weights/self.var_targets
        M = one_over_lambda * observations.T.dot(observations)
        M = np.linalg.inv(M + np.eye(M.shape[0]))

        mean = (one_over_lambda * M).dot(observations.T.dot(targets))
        cov = self.var_weights*M

        # Sample from posterior for Bayesian linear regression
        w = self.random_state.multivariate_normal(mean, cov)

        # Thompson sampling step -
        # Pull the arm with the highest prediction under the sampled model
        pred = np.zeros(trial.n_arms)
        for arm in range(trial.n_arms):
            feed = {self._istate: state, self._arms: [arm],
                    self._rewards: [trial.rewards[-1], 0]}

            arm_features = self.session.run(self._h_output, feed)[0]
            pred[arm] = arm_features.dot(w)

        if self.verbose:
            msg = 'Pull: {0:4d}. Loss: {1:.4f}. Prediction: {2}. Policy: {3}.'

            p = np.zeros(trial.n_arms)
            p[np.argmax(pred)] = 1.0

            with _printoptions(precision=4, suppress=True):
                if np.argmax(p) in self.bandit.best_arms():
                    cprint(msg.format(trial.length + 1, loss, pred, p), 'green')
                else:
                    print(msg.format(trial.length + 1, loss, pred, p))


        return np.argmax(pred)

    def __repr__(self):
        r = 'ThompsonRecurrentNetwork(n_units={0}, learning_rate={1}, '
        r += 'regL2={2}, epochs={3}, train_every={4}, std_targets={5}, \
        std_weights={6})'
        return r.format(self.n_units, self.learning_rate,
                        self.regularise_lambda, self.epochs,
                        self.train_every, np.sqrt(self.var_targets),
                        np.sqrt(self.var_weights))


class ThompsonFeedforwardNetwork(Policy):
    """ Neural-linear: Thompson sampling with Bayesian linear regression on
        the representation generated by a FFNN fed with handcrafted context. """

    def __init__(self, bandit, order, periods, n_units, learning_rate,
                 regularise_lambda, epochs, train_every, std_targets,
                 std_weights, verbose, seed):

        Policy.__init__(self, bandit)

        self.order = order
        self.periods = np.array(periods)
        self.n_units = n_units

        self.learning_rate = learning_rate
        self.regularise_lambda = regularise_lambda
        self.epochs = epochs
        self.train_every = train_every

        self.var_targets = std_targets**2.
        self.var_weights = std_weights**2.

        self.verbose = verbose

        self.random_state = np.random.RandomState(seed=seed)

        self._setup(seed)

    def _setup(self, seed):
        """ Creates the feedforward NN architecture used in neural-linear"""

        tf.reset_default_graph()
        tf.set_random_seed(seed)

        isize = len(self.periods)*2
        isize += self.bandit.n_arms + (self.bandit.n_arms + 1) * self.order

        self._observations = tf.placeholder(tf.float32, shape=[None, isize],
                                            name='observations')
        self._targets = tf.placeholder(tf.float32, shape=None, name='targets')

        n_units = [isize] + self.n_units
        self._h_output = self._observations
        for i in range(1, len(n_units)):
            W = tf.Variable(tf.truncated_normal(shape=(n_units[i - 1],
                                                       n_units[i])),
                            name='W{0}'.format(i))
            b = tf.Variable(tf.zeros(shape=(n_units[i])),
                            name='b{0}'.format(i))

            if i == 1:
                self._h_output = tf.matmul(self._h_output, W) + b
            else:
                self._h_output = tf.tanh(tf.matmul(self._h_output, W) + b)

        # Note: No bias in the output layer
        W = tf.Variable(tf.truncated_normal(shape=(n_units[-1], 1)),
                        name='Wout')

        self._pred = tf.matmul(self._h_output, W)
        self._pred = tf.reshape(self._pred, (-1,))

        # L2 regularization on weights
        self._reg_loss = sum(tf.nn.l2_loss(tf_var)
                for tf_var in tf.trainable_variables()
                if not ("b" in tf_var.name))

        self._loss = tf.reduce_mean((self._pred - self._targets)**2) + \
                    self.regularise_lambda * self._reg_loss

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self._train = optimizer.minimize(self._loss)

        config = tf.ConfigProto(device_count={'GPU': 0})
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())


    def _example(self, arms, rewards, t):
        """ Creates the handcrafted context-target for the Neural-linear
            model. """

        if t < self.order:
            raise Exception('Incomplete observation.')

        arms_oh = np.eye(self.bandit.n_arms)

        observation = []
        for p in self.periods:
            angle = (2*np.pi*t)/p
            observation += [np.cos(angle), np.sin(angle)]

        observation.extend(arms_oh[arms[t]])

        for i in range(self.order):
            observation.extend(arms_oh[arms[t - i - 1]])
            observation.append(rewards[t - i - 1])

        target = rewards[t]

        return observation, target

    def select(self, trial):
        """ Selects which arm to play in the current round according to Thompson
            sampling. """

        if trial.length < max(self.order + 1, trial.n_arms):
            return np.argmin(trial.pulls)

        observations = []
        targets = []
        for t in range(self.order, trial.length):
            observation, target = self._example(trial.arms, trial.rewards, t)

            observations.append(observation)
            targets.append(target)

        feed = {self._observations: observations, self._targets: targets}

        if trial.length % self.train_every == 0:
            # Train the NN weights when this condition is true
            for _ in range(self.epochs):
                self.session.run(self._train, feed)

        loss, observations = self.session.run([self._loss, self._h_output],
                                              feed)

        # Update the posterior for Bayesian linear regression
        one_over_lambda = self.var_weights/self.var_targets
        M = one_over_lambda * observations.T.dot(observations)
        M = np.linalg.inv(M + np.eye(M.shape[0]))

        mean = (one_over_lambda * M).dot(observations.T.dot(targets))
        cov = self.var_weights*M

        # Sample from posterior for Bayesian linear regression
        w = self.random_state.multivariate_normal(mean, cov)

        # Thompson sampling step -
        # Pull the arm with the highest prediction under the sampled model
        pred = np.zeros(trial.n_arms)
        for arm in range(trial.n_arms):
            observation, _ = self._example(trial.arms + [arm],
                                           trial.rewards + [0.],
                                           trial.length)
            feed = {self._observations: [observation]}
            arm_features = self.session.run(self._h_output, feed)[0]
            pred[arm] = arm_features.dot(w)

        if self.verbose:
            msg = 'Pull: {0:4d}. Loss: {1:.4f}. Prediction: {2}. Policy: {3}.'

            p = np.zeros(trial.n_arms)
            p[np.argmax(pred)] = 1.0

            with _printoptions(precision=4, suppress=True):
                if np.argmax(p) in self.bandit.best_arms():
                    cprint(msg.format(trial.length + 1, loss, pred, p), 'green')
                else:
                    print(msg.format(trial.length + 1, loss, pred, p))


        return np.argmax(pred)

    def __repr__(self):
        r = 'ThompsonFeedforwardNetwork(order={0}, periods={1}, '
        r += 'n_units={2}, learning_rate={3}, regL2={4}, epochs={5}, \
        train_every={6}, '
        r += 'std_targets={7}, std_weights={8})'
        return r.format(self.order, self.periods, self.n_units,
                        self.learning_rate, self.regularise_lambda,
                        self.epochs, self.train_every,
                        np.sqrt(self.var_targets), np.sqrt(self.var_weights))


policies = {'Oracle': Oracle,
            'Fixed': Fixed,
            'Random': Random,
            'EpsilonGreedy': EpsilonGreedy,
            'Boltzmann': Boltzmann,
            'UCB': UCB,
            'ThompsonSamplingBernoulli': ThompsonSamplingBernoulli,
            'BoltzmannRecurrentNetwork': BoltzmannRecurrentNetwork,
            'ThompsonRecurrentNetwork': ThompsonRecurrentNetwork,
            'ThompsonFeedforwardNetwork': ThompsonFeedforwardNetwork}
