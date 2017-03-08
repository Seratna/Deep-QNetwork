from collections import deque
from random import sample

import tensorflow as tf
import numpy as np
import gym

from neural_network import NeuralNetwork
from models import Model


class DeepQNetwork(object):
    """

    """
    def __init__(self, net_structure: list, exp_length, gamma):
        self.num_observations = net_structure[0]
        self.num_actions = net_structure[-1]
        self.gamma = gamma
        self.epsilon = 1

        #
        self.min_epsilon = 0.1
        self.delta__epsilon = 0.001
        self.learn_count = 0
        self.update_freq = 100

        self.experience = deque(maxlen=exp_length)  # stored past experience for training

        self.q_network = NeuralNetwork(net_structure, scope='q_network')
        self.q_target_network = NeuralNetwork(net_structure, scope='target_network')

    def choose_action(self, observation):
        if np.random.uniform() > self.epsilon:
            assert observation.size == self.num_observations
            x = observation.reshape([1, self.num_observations])
            q_values = self.q_network.output.eval(feed_dict={self.q_network.x: x})  # use the q-net to estimate Q-values
            return np.argmax(q_values)  # return the action corresponds to the highest estimated Q-value
        else:  # random choose
            return np.random.randint(0, self.num_actions)

    def record(self, observation, action, reward, next_observation):
        self.experience.append((observation, action, reward, next_observation))

    def learn(self, batch_size, session):
        if self.learn_count % self.update_freq == 0:
            self.q_target_network.copy_network(self.q_network, session=session)
            print('q_target_network updated @ learn_count ', self.learn_count)

        batch = sample(self.experience, batch_size)

        state = np.array([exp[0] for exp in batch], dtype=np.float64)  # φ_t
        action = np.array([exp[1] for exp in batch], dtype=np.int32)
        reward = np.array([exp[2] for exp in batch], dtype=np.float64)
        next_state = np.array([exp[3] for exp in batch], dtype=np.float64)  # φ_{t+1}

        # compute max Q from the next states
        q_next = self.q_target_network.output.eval(feed_dict={self.q_target_network.x: next_state})  # Q(φ_{t+1}, a'; θ)
        target_reward = reward + np.max(q_next, axis=1) * self.gamma

        # build q_target
        q_target = self.q_network.output.eval(feed_dict={self.q_network.x: state})
        for row, a, r in zip(q_target, action, target_reward):
            row[a] = r

        # training
        loss = self.q_network.train(state, q_target, session=session)

        # update
        self.epsilon = max(self.min_epsilon, self.epsilon - self.delta__epsilon)
        self.learn_count += 1


def main():
    dqn = DeepQNetwork([4, 50, 50, 2],  # structure of neural net
                       exp_length=3000,  # size of experience pool
                       gamma=0.7)  # discount factor

    env = gym.make('CartPole-v0')
    step_count = 0

    with tf.Session() as sess:
        Model.start_new_session(sess)

        for i_episode in range(10000):
            observation = env.reset()
            print('episode', i_episode)

            while True:
                env.render()

                action = dqn.choose_action(observation)
                new_observation, _, done, info = env.step(action)

                # compute reward
                x, x_dot, theta, theta_dot = new_observation
                r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
                reward = r1 + r2

                # record
                dqn.record(observation, action, reward, new_observation)

                # learn
                if step_count>1000:
                    dqn.learn(batch_size=320, session=sess)

                step_count += 1
                observation = new_observation

                if done:
                    break


if __name__ == '__main__':
    main()