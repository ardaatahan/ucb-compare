import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ucb import *


num_arms = 10
num_iters = 50000

ucb1_rewards = np.zeros(num_iters)
ucb1 = ucb1(num_arms, num_iters)

ucb1.run()
ucb1_rewards = ucb1.rewards

action_counts = ucb1.arm_step_counts / num_iters * 100
df = pd.DataFrame(np.vstack([action_counts.reshape(-1, 1).T.round(2),
                             ucb1.reward_means.reshape(-1, 1).T.round(2)]),
                  index=["UCB1", "Expected Reward"],
                  columns=["action_idx = " + str(i) for i in range(num_arms)])
df.to_csv("out.csv")

plt.figure(figsize=(12, 8))
plt.plot(ucb1_rewards, label="UCB1")
plt.legend(bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.savefig("out.png")
