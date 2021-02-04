import numpy as np
import gpflow


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
        self.y = reward_distribution
        self.x = np.arange(num_arms, step=1.0).reshape(num_arms, 1)
        self.obs_x = np.zeros(num_iters).reshape(num_iters, 1)
        self.obs_y = np.zeros(num_iters).reshape(num_iters, 1)
        self.cumulative_regret = 0
        self.regrets = np.zeros(num_iters)

    def _ucb(self, mean, variance, beta=1):
        return np.argmax(mean + np.sqrt(beta) * variance)

    def run(self):
        for i in range(self.num_iters):
            kernel = gpflow.kernels.Matern52()
            model = gpflow.models.GPR(
                data=(self.obs_x, self.obs_y), kernel=kernel)
            optimizer = gpflow.optimizers.Scipy()
            opt_logs = optimizer.minimize(
                model.training_loss, model.trainable_variables, options=dict(maxiter=100))

            mean, variance = model.predict_f(self.x)
            x_t = self._ucb(mean, variance)

            print(x_t, np.max(self.y), self.y[x_t])

            self.cumulative_regret += np.max(self.y) - self.y[x_t]
            self.regrets[i] = self.cumulative_regret

            self.obs_x[i] = self.x[x_t]
            self.obs_y[i] = self.y[x_t]
