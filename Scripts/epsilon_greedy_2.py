import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import pickle


class eg2_model(nn.Module):
    """Epsilon Greedy model with replay"""

    def __init__(self, n_obs, n_actions):
        """
        Args:
            n_obs (int): Dimensions of the state space (int for this project)
            n_actions (int): Number of possible actions
            lr (float, optional): Learning rate for the network. Defaults to 5e-4.
        """  
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_obs, n_obs * 10),
            nn.ELU(),
            nn.Linear(n_obs * 10, n_actions)
        )

        self.n_actions = n_actions

        self.replay_memory = deque([], 500)

    def forward(self, state):
        """Predictions values given state

        Args:
            state (np.array): Observation for given step

        Returns:
            q_values
        """
        q_vals = self.net(state)
        return q_vals

    def get_action(self, state, epsilon=0):
        """ Sample actions with epsilon-greedy policy

        Args:
            state (np.array): Observation for a given step
            epsilon (float, optional): Exploration probability. Defaults to 0.

        Returns:
            int: Action to take (card to play)
        """
        state = torch.tensor(
            state, dtype=torch.float32)  
        q_values = self.forward(state) 

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
        """
        Args:
            net (eg_model class): eg_model class, describing a neural network
            env (Trumps env): Custom enviroment made (swap for gym envs for example)
            num_updates (int, optional): Number of games to train for. Defaults to 100000.
            num_obs (int, optional): State space. Defaults to 104.
            lr (float, optional): Learning rate. Defaults to 5e-4.
            is_cuda (bool, optional): Whether to use graphics card. Please keep to False
            epsilon (int, optional): Initial exploration rate. Defaults to 1.
            min_epsilon (float, optional): Min exploration rate. Defaults to 0.1.
        """
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

        # Ignore for now
        self.is_cuda = torch.cuda.is_available() and is_cuda

        """Environment"""
        self.env = env

        """Network"""
        self.net = net
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def train(self):
        """Trains for self.num_updates number of games
        """        

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
        '''Generate sessions ready to train
        
        Args:
            sessions (int, optional): Number of sessions to generate
            
            '''

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
        """Evaluates model performance with epsilon = 0 for 500 games
        """ 
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
            gamma=0.99):
        """Compute loss function according to bellman equation

        Args:
            states (np.array): History of states from the last game
            actions (np.array): History of actions from the last game
            rewards (np.array): History of rewards from the last game
            next_states (np.array): History of states shifted by 1, from the last game
            is_done (bool): History of whether the game was finished, from the last game
            gamma (float, optional): discount rate. Defaults to 0.99.

        Returns:
            [torch.tensor] : Loss function ready to back propagate
        """ 

        states = torch.tensor(
            states, dtype=torch.float32)  
        actions = torch.tensor(
            actions, dtype=torch.long)   
        rewards = torch.tensor(
            rewards, dtype=torch.float32)  
        next_states = torch.tensor(next_states, dtype=torch.float32)

        is_done = torch.tensor(
            is_done, dtype=torch.uint8)  

        predicted_qvalues = self.net.forward(states)

        predicted_qvalues_for_actions = predicted_qvalues[range(
            states.shape[0]), actions]

        predicted_next_qvalues = self.net.forward(next_states)

        max_next_state_value = torch.max(predicted_next_qvalues, dim=1)[
            0] 

        target_qvalues_for_actions = rewards + \
            torch.mul(gamma, max_next_state_value)

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a)
        # since s' doesn't exist
        target_qvalues_for_actions = torch.where(
            is_done==1, rewards, target_qvalues_for_actions)

        # mean squared error loss to minimize
        loss = torch.mean((predicted_qvalues_for_actions -
                           target_qvalues_for_actions) ** 2)

        return loss

    def save(self):
        """Saves model"""
        with open('Scripts/models/epsilon_greedy_2_model', 'wb') as f:
            pickle.dump(self, f)
        return
