import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ucb import *


num_arms = 10
num_iters = 50000
reward_distribution = np.random.normal(0, 1, num_arms).reshape(num_arms, 1)

ucb1 = ucb1(num_arms, num_iters, reward_distribution)

ucb1.run()
ucb1_regrets = ucb1.regrets

action_counts = ucb1.arm_step_counts / num_iters * 100
df = pd.DataFrame(np.vstack([action_counts.reshape(-1, 1).T.round(2),
                             ucb1.reward_distribution.reshape(-1, 1).T.round(2)]),
                  index=["UCB1", "Expected Reward"],
                  columns=["action_idx = " + str(i) for i in range(num_arms)])
df.to_csv("out.csv")

plt.figure(figsize=(12, 8))
plt.plot(ucb1_regrets, label="UCB1")
plt.legend(bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Iterations")
plt.ylabel("Cumulative Regret")
plt.savefig("out.png")
