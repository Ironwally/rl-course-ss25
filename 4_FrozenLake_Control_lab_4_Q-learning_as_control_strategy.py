import random
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import math

env = gym.make("FrozenLake-v1")
random.seed(0)
np.random.seed(0)

print("## Frozen Lake ##")

no_states = env.observation_space.n
no_actions = env.action_space.n
q_values = np.zeros((no_states, no_actions))
q_counter = np.zeros((no_states, no_actions))


def play_episode(q_values, epsilon, alpha, gamma):

    state, _ = env.reset(seed=0)
    done = False
    reward_state = [] # reward_state
    state_action = [] # state_action
    time_step = 0
    while not done:
        # Task 1: use q-values to implement epsilon-greedy (behavior policy)
        chance = []
        best_actions = np.amax(q_values[state])
        no_best_actions = np.sum(q_values[state] == best_actions)
        for a in range(no_actions):
            chance.append(epsilon/no_actions)
            if q_values[state][a] == best_actions:
                chance[a] += (1 - epsilon)/no_best_actions

        action = random.choices(list(range(no_actions)), weights=chance, k=1)[0]
        state_action.append((state, action))

        # Task 3: greedy policy as (target policy)
        # Task 3: update q-values with Q-learning 
        # update q-values of last state using current state, done action and reward gained
        if time_step>0: 
            q_values[state_action[time_step-1][0]][state_action[time_step-1][1]] += alpha*(reward + gamma*q_values[state_action[time_step][0]][state_action[time_step][1]] - q_values[state_action[time_step-1][0]][state_action[time_step-1][1]])

        state, reward, done, _, _ = env.step(action)
        reward_state.append(reward)

        time_step+=1
    return state_action, reward_state

def train_agent(epsilon, no_episodes, alpha, gamma):
    rewards = []
    for i in range(0, no_episodes):
        s_a, r_s = play_episode(q_values, epsilon, alpha, gamma)
        rewards.append(sum(r_s))
        if i % 1000 == 0: print(f"Episode {i}")
        # Task 2: update q-values with SARSA
        # Happens inside episode

    plot_data = np.cumsum(rewards)
    return plot_data

def main():
    no_episodes = 8000
    epsilons = [0.01, 0.1, 0.5, 1.0, 1.2, 0.001, 0.0001]
    alpha = 0.5
    gamma = 0.1 # discount factor
    
    plot_data = [train_agent(eps, no_episodes, alpha, gamma) for eps in epsilons]

    num_eps = len(plot_data)
    ncols = 2 if num_eps > 1 else 1
    nrows = math.ceil(num_eps / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 8))
    # Ensure axs is a flat iterable even when there's only one subplot.
    if num_eps == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    for ax, data, eps in zip(axs, plot_data, epsilons):
        ax.plot(range(no_episodes), data, label=f"epsilon = {eps}")
        ax.set_xlabel("No. of episodes")
        ax.set_ylabel("Cumulative Sum of Rewards")
        ax.legend()

    # Remove any unused subplots
    for ax in axs[len(plot_data):]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    for data, eps in zip(plot_data, epsilons):
        ax.plot(range(no_episodes), data, label=f"epsilon = {eps}")
    ax.set_xlabel("No. of episodes")
    ax.set_ylabel("Cumulative Sum of Rewards")
    ax.set_title("Effect of Different Epsilon on Cumulative Rewards")
    ax.legend()
    plt.show()

main()
