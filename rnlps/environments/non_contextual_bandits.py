"""
    Non-contextual bandit environments to evaluate performance.

"""

import numpy as np

class StationaryBernoulliBandit:
    def __init__(self, means, seed):
        self.means = np.array(means)
        self.random_state = np.random.RandomState(seed)

        if (max(self.means) > 1.) or (min(self.means) < 0.):
            raise Exception('Invalid parameters.')

        self.n_arms = len(self.means)

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        regret = np.max(self.means) - self.means[arm]

        return float(self.random_state.binomial(1, self.means[arm])), None, regret

    def expected_cumulative_rewards(self, trial_length):
        return np.cumsum(np.full(trial_length, max(self.means)))

    def best_arms(self):
        m = max(self.means)
        return [a for a in range(self.n_arms) if np.allclose(self.means[a], m)]

    def __repr__(self):
        return 'StationaryBernoulliBandit(means={0})'.format(repr(self.means))


class FlippingBernoulliBandit:
    def __init__(self, means, half_period, seed):
        self.means = self._means = np.array(means)
        self.half_period = half_period
        self.random_state = np.random.RandomState(seed)

        if (max(self.means) > 1.) or (min(self.means) < 0.):
            raise Exception('Invalid parameters.')

        self.n_arms = len(self.means)
        self.step = 0

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        reward = float(self.random_state.binomial(1, self.means[arm]))

        regret = np.max(self.means) - self.means[arm]

        self.step += 1

        # flip means if step is a multiple of half_period
        if (self.step % self.half_period) == 0:
            self.means = 1 - self.means


        return reward, None, regret

    def expected_cumulative_rewards(self, trial_length):
        a, b = np.max(self._means), np.max(1. - self._means)
        er = np.array([a]*self.half_period + [b]*self.half_period)
        er = np.tile(er, trial_length//len(er) + 1)

        return np.cumsum(er[0: trial_length])

    def best_arms(self):
        m = max(self.means)
        return [a for a in range(self.n_arms) if np.allclose(self.means[a], m)]

    def __repr__(self):
        r = 'FlippingBernoulliBandit(means={0}, half_period={1})'
        return r.format(repr(self._means), self.half_period)


class SinusoidalBernoulliBandit:
    def __init__(self, n_arms, step_size, seed):
        self.n_arms = n_arms
        self.step_size = step_size
        self.random_state = np.random.RandomState(seed)

        self.pos = 0
        self.offsets = np.array([(2*np.pi*i)/self.n_arms for i in range(self.n_arms)])

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        p = (1. + np.sin(self.pos + self.offsets))/2.
        reward = float(self.random_state.binomial(1, p[arm]))
        regret = np.max(p) - p[arm]

        self.pos += self.step_size

        return reward, None, regret

    def expected_cumulative_rewards(self, trial_length):
        best_rewards = []
        pos = 0
        offsets = np.array([(2*np.pi*i)/self.n_arms for i in range(self.n_arms)])
        for i in range(trial_length):
            p = (1. + np.sin(pos + offsets))/2.
            best_rewards.append(np.max(p))
            pos += self.step_size

        return np.cumsum(best_rewards)


    def best_arms(self):
        p = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            p[arm] = (1. + np.sin(self.pos + self.offsets[arm]))/2.

        m = max(p)
        return [a for a in range(self.n_arms) if np.allclose(p[a], m)]

    def __repr__(self):
        r = 'SinusoidalBernoulliBandit(n_arms={0}, step_size={1})'
        return r.format(self.n_arms, self.step_size)


class StationaryGaussianBandit:
    def __init__(self, means, std, seed):
        self.means = np.array(means)
        self.std = float(std)
        self.random_state = np.random.RandomState(seed)

        self.n_arms = len(self.means)

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        reward = self.random_state.normal(self.means[arm], self.std)
        regret = np.max(self.means) - self.means[arm]

        return reward, None, regret

    def expected_cumulative_rewards(self, trial_length):
        return np.cumsum(np.full(trial_length, max(self.means)))

    def best_arms(self):
        m = max(self.means)
        return [a for a in range(self.n_arms) if np.allclose(self.means[a], m)]

    def __repr__(self):
        r = 'StationaryGaussianBandit(means={0}, std={1})'
        return r.format(repr(self.means), self.std)


class FlippingGaussianBandit:
    def __init__(self, means, std, half_period, seed):
        self.means = self._means = np.array(means)
        self.std = float(std)
        self.half_period = half_period
        self.random_state = np.random.RandomState(seed)

        self.n_arms = len(self.means)
        self.step = 0

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        reward = self.random_state.normal(self.means[arm], self.std)
        regret = np.max(self.means) - self.means[arm]

        self.step += 1

        # flip means if step is a multiple of half_period
        if (self.step % self.half_period) == 0:
            self.means = -self.means

        return reward, None, regret

    def expected_cumulative_rewards(self, trial_length):
        a, b = np.max(self._means), np.max(-self._means)
        er = np.array([a]*self.half_period + [b]*self.half_period)
        er = np.tile(er, trial_length//len(er) + 1)

        return np.cumsum(er[0: trial_length])

    def best_arms(self):
        m = max(self.means)
        return [a for a in range(self.n_arms) if np.allclose(self.means[a], m)]

    def __repr__(self):
        r = 'FlippingGaussianBandit(means={0}, std={1}, half_period={2})'
        return r.format(repr(self._means), self.std, self.half_period)


class SinusoidalGaussianBandit:
    def __init__(self, n_arms, std, step_size, seed):
        self.n_arms = n_arms
        self.std = float(std)
        self.step_size = step_size
        self.random_state = np.random.RandomState(seed)

        self.pos = 0
        self.offsets = np.array([(2*np.pi*i)/self.n_arms for i in range(self.n_arms)])

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        v = np.sin(self.pos + self.offsets)
        reward = self.random_state.normal(v[arm], self.std)
        regret = max(v) - v[arm]

        self.pos += self.step_size

        return reward, None, regret

    def expected_cumulative_rewards(self, trial_length):
        raise NotImplementedError()

    def best_arms(self):
        v = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            v[arm] = np.sin(self.pos + self.offsets[arm])

        m = max(v)
        return [a for a in range(self.n_arms) if np.allclose(v[a], m)]

    def __repr__(self):
        r = 'SinusoidalGaussianBandit(n_arms={0}, std={1}, step_size={2})'
        return r.format(self.n_arms, self.std, self.step_size)

class GaussianMarkovChainBandit:
    """ Most rewarding arm with expected reward = best_mean. All other arms
        return an expected reward = other_mean. When the most rewarding arm is
        pulled, it transitions according to the transition matrix (tmatrix)
        probabilities. """

    def __init__(self, tmatrix, best_mean, other_mean, std, seed):
        self.tmatrix = np.array(tmatrix)
        self.random_state = np.random.RandomState(seed)

        if self.tmatrix.shape[0] != self.tmatrix.shape[1]:
            raise Exception('Invalid matrix dimensions.')
        if not np.allclose(self.tmatrix.sum(axis=1), 1.):
            raise Exception('Invalid transition probabilities.')

        self.n_arms = self.tmatrix.shape[0]
        self.state = self.random_state.choice(self.n_arms)
        self.best_mean = best_mean
        self.other_mean = other_mean
        self.std = std

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        if (arm == self.state):
            p = self.tmatrix[self.state]
            self.state = self.random_state.choice(self.n_arms, p=p)
            reward_mu = self.best_mean
        else:
            reward_mu = self.other_mean

        reward = self.random_state.normal(reward_mu, self.std)
        regret = self.best_mean - reward_mu

        return reward, None, regret

    def expected_cumulative_rewards(self, trial_length):
        return np.cumsum(self.best_mean * np.ones(trial_length))

    def best_arms(self):
        return [self.state]

    def __repr__(self):
        r = 'GaussianMarkovChainBandit(tmatrix={0}, best_mean={1}, other_mean={2}, std={3})'
        return r.format(repr(self.tmatrix), self.best_mean, self.other_mean, self.std)

class GaussianCircularChainBandit(GaussianMarkovChainBandit):
    def __init__(self, n_arms, best_mean, other_mean, std, p_left, seed):
        if n_arms < 3:
            raise Exception('Invalid number of states.')

        tmatrix = np.zeros((n_arms, n_arms))
        for j in range(n_arms):
            tmatrix[j, j - 1] = p_left
            tmatrix[j, (j + 1) % n_arms] = (1 - p_left)

        GaussianMarkovChainBandit.__init__(self, tmatrix, best_mean, other_mean, std, seed)


non_contextual_bandits = {'StationaryBernoulliBandit': StationaryBernoulliBandit,
           'FlippingBernoulliBandit': FlippingBernoulliBandit,
           'SinusoidalBernoulliBandit': SinusoidalBernoulliBandit,
           'StationaryGaussianBandit': StationaryGaussianBandit,
           'FlippingGaussianBandit': FlippingGaussianBandit,
           'SinusoidalGaussianBandit': SinusoidalGaussianBandit,
           'GaussianMarkovChainBandit': GaussianMarkovChainBandit,
           'GaussianCircularChainBandit': GaussianCircularChainBandit}
