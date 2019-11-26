""" Bandit environments to evaluate performance. """
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path


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

        return float(self.random_state.binomial(1, self.means[arm]))

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

        self.step += 1

        if (self.step % self.half_period) == 0:
            self.means = 1 - self.means

        return reward

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
        self.offsets = [(2*np.pi*i)/self.n_arms for i in range(self.n_arms)]

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        p = (1. + np.sin(self.pos + self.offsets[arm]))/2.
        reward = float(self.random_state.binomial(1, p))

        self.pos += self.step_size

        return reward

    def expected_cumulative_rewards(self, trial_length):
        raise NotImplementedError()

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

        return self.random_state.normal(self.means[arm], self.std)

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

        self.step += 1

        if (self.step % self.half_period) == 0:
            self.means = -self.means

        return reward

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
        self.offsets = [(2*np.pi*i)/self.n_arms for i in range(self.n_arms)]

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        v = np.sin(self.pos + self.offsets[arm])
        reward = self.random_state.normal(v, self.std)

        self.pos += self.step_size

        return reward

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


class MarkovChainBandit:
    """ Single rewarding arm with reward = 1. All other arms return reward 0.
        When the rewarding arm is pulled, it transitions according to the
        transition matrix (tmatrix) probabilities. """

    def __init__(self, tmatrix, seed):
        self.tmatrix = np.array(tmatrix)
        self.random_state = np.random.RandomState(seed)

        if self.tmatrix.shape[0] != self.tmatrix.shape[1]:
            raise Exception('Invalid matrix dimensions.')
        if not np.allclose(self.tmatrix.sum(axis=1), 1.):
            raise Exception('Invalid transition probabilities.')

        self.n_arms = self.tmatrix.shape[0]
        self.state = self.random_state.choice(self.n_arms)

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        reward = float(arm == self.state)
        if reward > 0.:
            p = self.tmatrix[self.state]
            self.state = self.random_state.choice(self.n_arms, p=p)

        return reward

    def expected_cumulative_rewards(self, trial_length):
        raise NotImplementedError()

    def best_arms(self):
        return [self.state]

    def __repr__(self):
        r = 'MarkovChainBandit(tmatrix={0})'
        return r.format(repr(self.tmatrix))

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
            reward = self.random_state.normal(self.best_mean, self.std)
        else:
            reward = self.random_state.normal(self.other_mean, self.std)

        return reward

    def expected_cumulative_rewards(self, trial_length):
        return np.cumsum(self.best_mean * np.ones(trial_length))

    def best_arms(self):
        return [self.state]

    def __repr__(self):
        r = 'MarkovChainBandit(tmatrix={0})'
        return r.format(repr(self.tmatrix))


class DelayedMarkovChainBandit:
    def __init__(self, tmatrix, r_prob, seed):
        self.tmatrix = np.array(tmatrix)
        self.random_state = np.random.RandomState(seed)

        if self.tmatrix.shape[0] != self.tmatrix.shape[1]:
            raise Exception('Invalid matrix dimensions.')
        if not np.allclose(self.tmatrix.sum(axis=1), 1.):
            raise Exception('Invalid transition probabilities.')

        self.dist_matrix = shortest_path((self.tmatrix > 0).astype(float))

        self.n_arms = self.tmatrix.shape[0]
        self.state = self.random_state.choice(self.n_arms)
        self.acc_reward = 0.0
        self.r_prob = r_prob

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        observed_reward = 0.0

        #reward = float(arm == self.state)
        reward = np.exp(-1 * np.float32(self.dist_matrix[arm, self.state]))
        self.acc_reward = self.acc_reward + reward

        if reward == 1:
            p = self.tmatrix[self.state]
            self.state = self.random_state.choice(self.n_arms, p=p)


        if self.random_state.binomial(1, self.r_prob):
            observed_reward = self.acc_reward
            self.acc_reward = 0.0

        return observed_reward

    def expected_cumulative_rewards(self, trial_length):
        raise NotImplementedError()

    def best_arms(self):
        return [self.state]

    def __repr__(self):
        r = 'DelayedMarkovChainBandit(tmatrix={0})'
        return r.format(repr(self.tmatrix))


class CircularChainBandit(MarkovChainBandit):

    def __init__(self, n_arms, p_left, seed):
        if n_arms < 3:
            raise Exception('Invalid number of states.')

        tmatrix = np.zeros((n_arms, n_arms))
        for j in range(n_arms):
            tmatrix[j, j - 1] = p_left
            tmatrix[j, (j + 1) % n_arms] = (1 - p_left)

        MarkovChainBandit.__init__(self, tmatrix, seed)


class GaussianCircularChainBandit(GaussianMarkovChainBandit):
    def __init__(self, n_arms, best_mean, other_mean, std, p_left, seed):
        if n_arms < 3:
            raise Exception('Invalid number of states.')

        tmatrix = np.zeros((n_arms, n_arms))
        for j in range(n_arms):
            tmatrix[j, j - 1] = p_left
            tmatrix[j, (j + 1) % n_arms] = (1 - p_left)

        GaussianMarkovChainBandit.__init__(self, tmatrix, best_mean, other_mean, std, seed)


class DelayedCircularChainBandit(DelayedMarkovChainBandit):
    def __init__(self, n_arms, p_left, r_prob, seed):
        if n_arms < 3:
            raise Exception('Invalid number of states.')

        tmatrix = np.zeros((n_arms, n_arms))
        for j in range(n_arms):
            tmatrix[j, j - 1] = p_left
            tmatrix[j, (j + 1) % n_arms] = (1 - p_left)

        DelayedMarkovChainBandit.__init__(self, tmatrix, r_prob, seed)


class DistanceBandit:
    def __init__(self, adj_matrix, reward_sigma, seed):
        self.random_state = np.random.RandomState(seed + 1)

        self.adj_matrix = np.array(adj_matrix)
        if not np.allclose(self.adj_matrix, self.adj_matrix.T):
            raise Exception('Invalid adjacency matrix.')
        self.dist_matrix = shortest_path(self.adj_matrix)

        self.n_arms = self.adj_matrix.shape[0]
        self.state = self.random_state.choice(self.n_arms)

        self.reward_sigma = reward_sigma


    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        dist = np.float32(self.dist_matrix[arm, self.state])
        reward_mean = np.exp(-1 * dist)
        reward = self.random_state.normal(reward_mean, self.reward_sigma)


        if reward_mean == 1:
            self.state = self.random_state.choice(self.n_arms)

        return reward

    def expected_cumulative_rewards(self, trial_length):
        return np.cumsum(np.ones(trial_length))

    def best_arms(self):
        return [self.state]


class RandomDistanceBandit(DistanceBandit):
    def __init__(self, n_arms, p, reward_sigma, seed):

        G = nx.gnp_random_graph(n_arms, p=p, seed=seed)

        tmatrix = nx.to_numpy_array(G)

        DistanceBandit.__init__(self, tmatrix, reward_sigma, seed)


class RandomChainBandit(MarkovChainBandit):
    def __init__(self, n_arms, alpha, seed):
        random_state = np.random.RandomState(seed + 1)
        tmatrix = random_state.dirichlet([alpha]*n_arms, n_arms)

        MarkovChainBandit.__init__(self, tmatrix, seed)


class RandomPermutationBandit(MarkovChainBandit):
    def __init__(self, n_arms, seed):
        random_state = np.random.RandomState(seed + 1)
        perm = random_state.permutation(n_arms)
        imatrix = np.eye(n_arms)
        tmatrix = imatrix[perm]

        while sum(np.diag(tmatrix) > 0):
            perm = random_state.permutation(n_arms)
            tmatrix = imatrix[perm]

        MarkovChainBandit.__init__(self, tmatrix, seed)


class NeighborhoodBandit:
    def __init__(self, adj_matrix):
        self.adj_matrix = np.array(adj_matrix)
        if not np.allclose(self.adj_matrix, self.adj_matrix.T):
            raise Exception('Invalid adjacency matrix.')
        self.dist_matrix = shortest_path(self.adj_matrix)

        self.n_arms = self.adj_matrix.shape[0]
        self.state = np.full(self.n_arms, 1./self.n_arms)

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        reward = self.state[arm]

        dist = np.exp(-(self.dist_matrix[arm]**2))
        dist = dist/np.sum(dist)

        self.state = self.state*(1 - dist)
        self.state = self.state/np.sum(self.state)

        return reward

    def expected_cumulative_rewards(self, trial_length):
        raise NotImplementedError()

    def best_arms(self):
        m = np.max(self.state)
        return [a for a in range(self.n_arms) if np.allclose(self.state[a], m)]

    def __repr__(self):
        r = 'NeighborhoodBandit(adj_matrix={0})'
        return r.format(self.adj_matrix)


class CircularNeighborhoodBandit(NeighborhoodBandit):
    def __init__(self, n_arms):
        tmatrix = np.zeros((n_arms, n_arms))
        for j in range(n_arms):
            tmatrix[j, j - 1] = 1
            tmatrix[j, (j + 1) % n_arms] = 1

        NeighborhoodBandit.__init__(self, tmatrix)


class RandomNeighborhoodBandit(NeighborhoodBandit):
    def __init__(self, n_arms, seed):
        random_state = np.random.RandomState(seed)

        tmatrix = np.triu(random_state.randint(2, size=(n_arms, n_arms)))
        tmatrix = np.minimum(1, tmatrix + tmatrix.T)

        NeighborhoodBandit.__init__(self, tmatrix)


bandits = {'StationaryBernoulliBandit': StationaryBernoulliBandit,
           'FlippingBernoulliBandit': FlippingBernoulliBandit,
           'SinusoidalBernoulliBandit': SinusoidalBernoulliBandit,
           'StationaryGaussianBandit': StationaryGaussianBandit,
           'FlippingGaussianBandit': FlippingGaussianBandit,
           'SinusoidalGaussianBandit': SinusoidalGaussianBandit,
           'MarkovChainBandit': MarkovChainBandit,
           'GaussianMarkovChainBandit': GaussianMarkovChainBandit,
           'DelayedMarkovChainBandit': DelayedMarkovChainBandit,
           'CircularChainBandit': CircularChainBandit,
           'GaussianCircularChainBandit': GaussianCircularChainBandit,
           'DelayedCircularChainBandit': DelayedCircularChainBandit,
           'DistanceBandit': DistanceBandit,
           'RandomDistanceBandit': RandomDistanceBandit,
           'RandomChainBandit': RandomChainBandit,
           'NeighborhoodBandit': NeighborhoodBandit,
           'CircularNeighborhoodBandit': CircularNeighborhoodBandit,
           'RandomNeighborhoodBandit': RandomNeighborhoodBandit}
