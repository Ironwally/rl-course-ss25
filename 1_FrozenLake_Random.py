# maximize return per episode
# Run multiple exact same episodes
# improve policy after each episode
# policy = probability for each action saved per each state
# actions = states that can be transitioned to from another state 
# return = estimation of sum of rewards to be collected until end of episode
# reward = reward for entering state
# state = situation the agent can be in at time t, defining possible actions, possible reward
# possible policy = change probability if one action gives higher result than other

import gymnasium as gym
import random


class Episode:
    def __init__(self):
        self.actions = []
        self.states = []
    def __repr__(self): 
        return "Episode actions:%s (%s) states:%s (%s)" % (self.actions, len(self.actions), self.states, len(self.states))


env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi") # ansi or human

random.seed(0)

num_actions = 4 # in all states
num_states = 16

environment_states = [[0.25]*num_actions]*num_states

def policy_improver(episode):
    for i, state in enumerate(episode.states):
        action = environment_states[state][episode.actions[i]]
        #environment_states[state][episode.actions[i]] = environment_states[#state that action leads to#]

print("## Frozen Lake ##")

action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

for i in range(0,10):
    episode_count = 0
    reward = 0
    while not reward>0:
        episode_count+=1
        episode_done = False
        state = env.reset(seed=0)
        state = 0
        print("Start state:")
        print(state)
        print(env.render()) 
        episode = Episode()
        while not episode_done:
            episode.states.append(state)
            action = random.choices(range(num_actions), weights=environment_states[state])[0]
            episode.actions.append(action)
            state, reward, episode_done, _, _ = env.step(action)
            print(f"\nAction:{action2string[action]}, new state:{state}, reward:{reward}")
            print(env.render())
        print("-----------------------------------")
        policy_improver(episode)


    print(f"Episode count: {episode_count}")
    print(episode)
    #policy_improver(episode)

    print("#################################")