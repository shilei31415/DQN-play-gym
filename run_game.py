import gym
from RL_brain import DeepQNetwork
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ep_score=[]
def plot_ep(ep):
    ep_score.append(ep)
    plt.plot(np.arange(len(ep_score)), ep_score)
    plt.ylabel('ep')
    plt.xlabel('training steps')
    plt.draw()
    plt.pause(0.01)


env = gym.make("SpaceInvaders-v0")
env = env.unwrapped

print(env.action_space)
print(env.observation_space.shape)
# print(env.observation_space.high)
# print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  shape_features=env.observation_space.shape,
                  learning_rate=0.01, e_greedy=0.95,
                  replace_target_iter=1000, memory_size=5000,
                  e_greedy_increment=0.00001)

def update(step):
    # RL.sess.run(RL.replace_target_op)
    # print("\ntarget_params_replaced\n")
    if RL.epsilon<0.8:
        RL.sess.run(RL.replace_target_op)
        print("\ntarget_params_replaced\n")
    else:
        print("新旧网络对比")
        ep_r_eval = 0
        for i in tqdm(range(10)):
            observation = env.reset()
            while True:
                env.render()

                action = RL.choose_action_eval(observation)

                observation_, reward, done, info = env.step(action)

                observation = observation_

                ep_r_eval += reward
                if done:
                    break

        ep_r_next = 0
        for i in tqdm(range(10)):
            observation = env.reset()
            while True:
                env.render()

                action = RL.choose_action_target(observation)

                observation_, reward, done, info = env.step(action)

                observation = observation_

                ep_r_next += reward
                if done:
                    break
        print("新网络：", ep_r_eval)
        print("旧网络：", ep_r_next)
        if ep_r_eval>=ep_r_next:
            RL.sess.run(RL.replace_target_op)
            print("\ntarget_params_replaced\n")

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
            if total_steps > 1000:
                if RL.learn_step_counter % RL.replace_target_iter == 0:
                    update(total_steps)
                RL.learn()
                # pass

            if done:
                print('episode: ', i_episode,
                      'ep_r: ', round(ep_r, 2),
                      ' epsilon: ', round(RL.epsilon, 2))
                plot_ep(ep_r)
                break

            observation = observation_
            total_steps += 1

    RL.plot_cost()

if __name__ == '__main__':
    main()