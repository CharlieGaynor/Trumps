import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import pickle


class eg2_model(nn.Module):

    def __init__(self, n_obs, n_actions):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_obs, n_obs * 10),
            nn.ELU(),
            nn.Linear(n_obs * 10, n_actions)
        )

        self.n_actions = n_actions

        self.replay_memory = deque([], 500)

    def forward(self, state):
        q_vals = self.net(state)
        return q_vals

    def get_action(self, state, epsilon=0):
        """
        sample actions with epsilon-greedy policy
        recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
        """
        state = torch.tensor(
            state, dtype=torch.float32)  # None adds a new axis, why do we need it ?
        q_values = self.forward(state)  # .detach() #this may be wrong

        ran_num = torch.rand(1)
        if ran_num < epsilon:
            return int(torch.randint(low=0, high=self.n_actions, size=(1,)))

        else:
            return int(torch.argmax(q_values))


class eg2_runner(object):

    def __init__(
            self,
            net,
            env,
            num_updates=100000,
            num_obs=104,
            lr=5e-4,
            is_cuda=False,
            epsilon=1,
            min_epsilon=0.1):
        super().__init__()

        # constants
        self.num_obs = num_obs
        self.lr = lr

        self.num_updates = num_updates
        self.best_loss = np.inf
        self.sum_rewards = []
        self.replay_memory = deque([], 1000)
        self.epsilon = epsilon
        self.max_epsilon = 1
        self.min_epsilon = min_epsilon
        # change epsilon so we end up at min_epsilon after #num_updates
        self.ep_multiplier = self.min_epsilon**(1 / num_updates)
        self.scores = []

        # loss scaling coefficients
        self.is_cuda = torch.cuda.is_available() and is_cuda

        """Environment"""
        self.env = env

        """Network"""
        self.net = net
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def train(self):

        for chunk in range(1, self.num_updates // 250 + 1):

            self.generate_sessions(500)
            memories = random.sample(self.replay_memory, 250)

            for s, a, r, next_s, done in memories:
                self.opt.zero_grad()
                loss = self.compute_td_loss([s], [a], [r], [next_s], [done])
                loss.backward()
                for param in self.net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.opt.step()

            if chunk % 2 == 0:
                print('.', end='')
            if chunk % 25 == 0:
                self.evaluate()
            if chunk % 100 == 0:
                print(f'{chunk*250}')

        return

    def generate_sessions(self, sessions=500):
        '''Generate sessions ready to train'''

        for _ in range(1, sessions):
            s, pc = self.env.reset()
            done = False
            total_reward = 0
            self.epsilon = self.epsilon * self.ep_multiplier
            while not done:
                a = self.net.get_action(s, epsilon=self.epsilon)
                next_s, r, done, pc = self.env.step(a)
                total_reward += r
                self.replay_memory.append([s, a, r, next_s, done])
                s = next_s
                if done:
                    self.sum_rewards.append(total_reward)
                    break

        return

    def evaluate(self):
        for _ in range(500):
            total_reward = 0
            s, pc = self.env.reset()
            for __ in range(100):
                a = self.net.get_action(s, epsilon=0)
                next_s, r, done, pc = self.env.step(a)
                total_reward += r
                s = next_s
                if done:
                    self.scores.append(total_reward)
                    break
        return

    def compute_td_loss(
            self,
            states,
            actions,
            rewards,
            next_states,
            is_done,
            gamma=0.99,
            check_shapes=False):
        """ Compute td loss"""

        states = torch.tensor(
            states, dtype=torch.float32)    # shape: [batch_size, state_size]
        actions = torch.tensor(
            actions, dtype=torch.long)   # shape: [batch_size]
        rewards = torch.tensor(
            rewards, dtype=torch.float32)  # shape: [batch_size]
        next_states = torch.tensor(next_states, dtype=torch.float32)

        is_done = torch.tensor(
            is_done, dtype=torch.uint8)  # shape: [batch_size]

        predicted_qvalues = self.net.forward(states)

        predicted_qvalues_for_actions = predicted_qvalues[range(
            states.shape[0]), actions]

        predicted_next_qvalues = self.net.forward(next_states)

        max_next_state_value = torch.max(predicted_next_qvalues, dim=1)[
            0]  # .to(self.device)

        target_qvalues_for_actions = rewards + \
            torch.mul(gamma, max_next_state_value)

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a)
        # since s' doesn't exist
        target_qvalues_for_actions = torch.where(
            is_done, rewards, target_qvalues_for_actions)

        # mean squared error loss to minimize
        loss = torch.mean((predicted_qvalues_for_actions -
                           target_qvalues_for_actions) ** 2)

        return loss

    def save(self):
        with open('models/epsilon_greedy_2_model', 'wb') as f:
            pickle.dump(self, f)
        return
