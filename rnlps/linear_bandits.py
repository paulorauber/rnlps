"""
    Linear bandit environments to evaluate performance.

"""

import numpy as np
import os

class StationaryLinearBandit:

    def __init__(self, n_arms, dimension, seed, arm_pool_size = 2000, err_sigma = 0.05):

        self.n_arms = n_arms
        self.dimension = dimension
        self.arm_pool_size = arm_pool_size
        self.err_sigma = err_sigma

        self.random_state = np.random.RandomState(seed)

        self.theta_star = self.generate_theta_star()
        self.arm_pool = self.generate_arm_pool()

        self.current_arms = []

        self.step = 0

    def generate_theta_star(self):
        theta_star_unnormalized = self.random_state.uniform(low = -1, high = 1, size = (self.dimension,))
        return theta_star_unnormalized/np.linalg.norm(theta_star_unnormalized)

    def generate_arm_pool(self):
        arm_pool_unnormalized = self.random_state.uniform(low = -1, high = 1, size = (self.arm_pool_size, self.dimension))
        return arm_pool_unnormalized/np.linalg.norm(arm_pool_unnormalized, keepdims = True, axis = 1)

    def sample_arms(self):
        indices = self.random_state.choice(self.arm_pool_size, size = self.n_arms, replace = False)
        return self.arm_pool[indices]

    def reset(self):
        self.step = 0
        arms_context = self.sample_arms()
        self.current_arms = arms_context
        return arms_context

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        expected_reward = np.dot(self.current_arms[arm], self.theta_star)

        best_arm = self.best_arms()
        regret = np.dot(self.current_arms[best_arm[0]] , self.theta_star) - expected_reward

        reward = expected_reward + self.random_state.normal(0, self.err_sigma)

        self.step += 1
        context = self.sample_arms()

        self.current_arms = context

        return reward, context, regret

    def best_arms(self):
        means = np.dot(self.current_arms, self.theta_star)
        return [np.argmax(means)]

    def expected_cumulative_rewards(self, trial_length):
        raise NotImplementedError

    def __repr__(self):
        r = 'StationaryLinearBandit(n_arms={0}, dimension={1}, arm_pool_size={2})'
        return r.format(self.n_arms, self.dimension, self.arm_pool_size)


class FlippingLinearBandit:

    def __init__(self, n_arms, dimension, half_period, seed, arm_pool_size = 2000, err_sigma = 0.05):

        self.n_arms = n_arms
        self.dimension = dimension
        self.arm_pool_size = arm_pool_size
        self.err_sigma = err_sigma
        self.half_period = half_period

        self.random_state = np.random.RandomState(seed)

        self.theta_star = self.generate_theta_star()
        self.arm_pool = self.generate_arm_pool()

        self.current_arms = []

        self.step = 0

    def generate_theta_star(self):
        theta_star_unnormalized = self.random_state.uniform(low = -1, high = 1, size = (self.dimension,))
        return theta_star_unnormalized/np.linalg.norm(theta_star_unnormalized)

    def generate_arm_pool(self):
        arm_pool_unnormalized = self.random_state.uniform(low = -1, high = 1, size = (self.arm_pool_size, self.dimension))
        return arm_pool_unnormalized/np.linalg.norm(arm_pool_unnormalized, keepdims = True, axis = 1)

    def sample_arms(self):
        indices = self.random_state.choice(self.arm_pool_size, size = self.n_arms, replace = False)
        return self.arm_pool[indices]

    def reset(self):
        self.step = 0
        arms_context = self.sample_arms()
        self.current_arms = arms_context
        return arms_context

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        expected_reward = np.dot(self.current_arms[arm], self.theta_star)

        best_arm = self.best_arms()
        regret = np.dot(self.current_arms[best_arm[0]] , self.theta_star) - expected_reward

        reward = expected_reward + self.random_state.normal(0, self.err_sigma)

        self.step += 1
        context = self.sample_arms()

        # Update theta_star
        if self.step % self.half_period == 0:
            self.theta_star = -1 * self.theta_star

        self.current_arms = context

        return reward, context, regret

    def best_arms(self):
        means = np.dot(self.current_arms, self.theta_star)
        return [np.argmax(means)]

    def expected_cumulative_rewards(self, trial_length):
        raise NotImplementedError

    def __repr__(self):
        r = 'FlippingLinearBandit(n_arms={0}, dimension={1}, half_period={2}, arm_pool_size={3})'
        return r.format(self.n_arms, self.dimension, self.half_period, self.arm_pool_size)


class RotatingLinearBandit2d:

    def __init__(self, n_arms, time_period, seed, arm_pool_size = 2000, err_sigma = 0.05):

        self.n_arms = n_arms
        self.dimension = 2
        self.arm_pool_size = arm_pool_size
        self.err_sigma = err_sigma

        self.time_period = time_period

        self.random_state = np.random.RandomState(seed)
        self.arm_pool = self.generate_arm_pool()

        self.current_arms = []

        self.step = 0
        self.theta_star = np.array([np.cos(2 * np.pi * self.step/self.time_period), np.sin(2 * np.pi * self.step/self.time_period)])

    def generate_arm_pool(self):
        arm_pool_unnormalized = self.random_state.uniform(low = -1, high = 1, size = (self.arm_pool_size, self.dimension))
        return arm_pool_unnormalized/np.linalg.norm(arm_pool_unnormalized, keepdims = True, axis = 1)

    def sample_arms(self):
        indices = self.random_state.choice(self.arm_pool_size, size = self.n_arms, replace = False)
        return self.arm_pool[indices]

    def reset(self):
        self.step = 0
        arms_context = self.sample_arms()
        self.current_arms = arms_context
        return arms_context

    def pull(self, arm):
        if (arm >= self.n_arms) or (arm < 0):
            raise Exception('Invalid arm.')

        expected_reward = np.dot(self.current_arms[arm], self.theta_star)

        best_arm = self.best_arms()
        regret = np.dot(self.current_arms[best_arm[0]] , self.theta_star) - expected_reward

        reward = expected_reward + self.random_state.normal(0, self.err_sigma)

        self.step += 1
        context = self.sample_arms()

        # Update theta_star
        self.theta_star = np.array([np.cos(2 * np.pi * self.step/self.time_period), np.sin(2 * np.pi * self.step/self.time_period)])

        self.current_arms = context

        return reward, context, regret

    def best_arms(self):
        means = np.dot(self.current_arms, self.theta_star)
        return [np.argmax(means)]

    def expected_cumulative_rewards(self, trial_length):
        raise NotImplementedError

    def __repr__(self):
        r = 'RotatingLinearBandit2d(n_arms={0}, dimension={1}, time_period={2}, arm_pool_size={3})'
        return r.format(self.n_arms, self.dimension, self.half_period, self.arm_pool_size)

linear_bandits = {'StationaryLinearBandit': StationaryLinearBandit,
            'FlippingLinearBandit': FlippingLinearBandit,
            'RotatingLinearBandit2d' : RotatingLinearBandit2d}
