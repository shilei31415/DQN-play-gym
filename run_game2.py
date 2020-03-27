import gym
from RL_brain2 import DeepQNetwork
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

env = gym.make("SpaceInvaders-ram-v0")
env = env.unwrapped

print(env.action_space)
print(env.observation_space.shape[0])
print(env.observation_space.high)
print(env.observation_space.low)

ep_score=[]
def plot_ep():
    plt.plot(np.arange(len(ep_score)), ep_score)
    plt.ylabel('ep')
    plt.xlabel('training steps')
    plt.draw()
    plt.pause(0.01)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=1000, memory_size=5000,
                  e_greedy_increment=0.000002)

def main():
    total_steps = 0
    for i_episode in range(100000):

        observation = env.reset()
        ep_r = 0
        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            ep_r += reward
            if total_steps > 5000 and total_steps % 5==0:
                RL.learn()
                # pass

            if done:
                print('episode: ', i_episode,
                      'ep_r: ', round(ep_r, 2),
                      ' epsilon: ', round(RL.epsilon, 2))
                # plot_ep(ep_r)

                break

            observation = observation_
            total_steps += 1
    # plot_ep()
    RL.plot_cost()

if __name__ == '__main__':
    main()
    plt.pause(400000)