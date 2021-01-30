import numpy as np


class ucb:
    def __init__(self, num_arms, num_iters):
        self.num_arms = num_arms
        self.num_iters = num_iters
        self.step_count = 1
        self.arm_step_counts = np.ones(num_arms)
        self.mean_reward = 0
        self.rewards = np.zeros(num_iters)
        self.arms_mean_reward = np.zeros(num_arms)
        self.reward_means = np.random.normal(0, 1, num_arms)

    def play(self):
        action_idx = np.argmax(
            self.rewards + np.sqrt(2 * np.log(self.step_count) / self.arm_step_counts))

        reward = np.random.normal(self.reward_means[action_idx], 1)

        self.step_count += 1
        self.arm_step_counts[action_idx] += 1

        self.mean_reward = self.mean_reward + \
            (reward - self.mean_reward) / self.step_count

        self.arms_mean_reward[action_idx] = self.arms_mean_reward[action_idx] + \
            (reward - self.arms_mean_reward[action_idx]) / self.step_count

    def run(self):
        for i in range(self.iters):
            self.play()
            self.rewards[i] = self.mean_reward

    def reset(self):
        self.step_count = 1
        self.arm_step_counts = np.ones(num_arms)
        self.mean_reward = 0
        self.rewards = np.zeros(num_iters)
        self.arms_mean_reward = np.zeros(num_arms)
        self.reward_means = np.random.normal(0, 1, num_arms)


class gp_ucb:
    pass
