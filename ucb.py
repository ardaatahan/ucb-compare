import numpy as np


class ucb1:
    def __init__(self, num_arms, num_iters, reward_distribution):
        self.num_arms = num_arms
        self.num_iters = num_iters
        self.step_count = 1
        self.arm_step_counts = np.ones(num_arms)
        self.mean_reward = 0
        self.rewards = np.zeros(num_iters)
        self.arms_mean_reward = np.zeros(num_arms)
        self.reward_distribution = reward_distribution
        self.cumulative_regret = 0
        self.regrets = np.zeros(num_iters)

    def _play(self):
        action_idx = np.argmax(
            self.arms_mean_reward + np.sqrt(2 * np.log(self.step_count) / self.arm_step_counts))

        reward = self.reward_distribution[action_idx]
        self.cumulative_regret += (np.max(self.reward_distribution) - reward)

        self.step_count += 1
        self.arm_step_counts[action_idx] += 1

        self.mean_reward = self.mean_reward + \
            (reward - self.mean_reward) / self.step_count
        self.arms_mean_reward[action_idx] = self.arms_mean_reward[action_idx] + \
            (reward - self.arms_mean_reward[action_idx]) / self.step_count

    def run(self):
        for i in range(self.num_iters):
            self._play()
            self.rewards[i] = self.mean_reward
            self.regrets[i] = self.cumulative_regret


class gp_ucb:
    def __init__(self, num_arms, num_iters, reward_distribution):
        self.num_arms = num_arms
        self.num_iters = num_iters
        self.mean_reward = 0
        self.rewards = np.zeros(num_iters)
        self.arms_mean_reward = np.zeros(num_arms)
        self.reward_distribution = reward_distribution

    def _ucb(mean, variance, beta):
        return np.argmax(mean + np.sqrt(beta) * variance)

    def play(self):
        pass

    def run(self):
        pass
