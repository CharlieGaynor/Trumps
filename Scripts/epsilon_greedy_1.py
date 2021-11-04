import torch
from torch import nn
import numpy as np
import pickle


class eg_model(nn.Module):
    """Epsilon Greedy model"""

    def __init__(self, n_obs, n_actions, lr=5e-4):
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
        ) #Simple network
        self.n_actions = n_actions
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

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
        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.forward(state)

        ran_num = torch.rand(1)
        if ran_num < epsilon:
            return int(torch.randint(low=0, high=self.n_actions, size=(1,)))
        else:
            return int(torch.argmax(q_values))


class eg_runner(object):

    def __init__(self, net, env, num_updates=100000, num_obs=104,
                 lr=5e-4, epsilon=1, min_epsilon=0.1):
        """
        Args:
            net (eg_model class): eg_model class, describing a neural network
            env (Trumps env): Custom enviroment made (swap for gym envs for example)
            num_updates (int, optional): Number of games to train for. Defaults to 100000.
            num_obs (int, optional): State space. Defaults to 104.
            lr (float, optional): Learning rate. Defaults to 5e-4.
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
        self.epsilon = epsilon
        self.max_epsilon = 1
        self.min_epsilon = min_epsilon
        self.ep_multiplier = self.min_epsilon**(1 / num_updates)
        self.scores = []
        # Using gpu? Set False for now
        self.is_cuda = False

        """Environment"""
        self.env = env

        """Network"""
        self.net = net

    def train(self, t_max=10, train=True):
        """Play env with approximate q-learning agent and train it at the same time

        Args:
            t_max (int, optional): Max steps in a game. Leave it at and Defaults to 10.
            train (bool, optional): Whether to train or not. Defaults to True.
        """        
        for episode in range(1, self.num_updates + 1):
            total_reward = 0
            s, pc = self.env.reset()

            # Plays a game
            for t in range(t_max):
                a = self.net.get_action(s, epsilon=self.epsilon)
                next_s, r, done, pc = self.env.step(a)

                if train:  # Train on what just happened
                    self.net.opt.zero_grad()
                    loss = self.compute_td_loss(
                        [s], [a], [r], [next_s], [done])
                    loss.backward()
                    self.net.opt.step()

                total_reward += r
                s = next_s
                if done:
                    self.sum_rewards.append(total_reward)
                    self.epsilon = self.epsilon * self.ep_multiplier
                    break

            # Print Stuff
            if (episode) % 20000 == 0:
                print(f"{episode}")
            if (episode) % 5000 == 0:
                self.evaluate()
            if (episode) % 400 == 0:
                print('.', end='')
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

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        is_done = torch.tensor(is_done, dtype=torch.uint8) 

        predicted_qvalues = self.net.forward(states)

        predicted_qvalues_for_actions = predicted_qvalues[range(states.shape[0]), actions]

        predicted_next_qvalues = self.net.forward(next_states)

        max_next_state_value = torch.max(predicted_next_qvalues, dim=1)[0]

        target_qvalues_for_actions = rewards + torch.mul(gamma, max_next_state_value)

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a)
        # since s' doesn't exist
        target_qvalues_for_actions = torch.where(
            is_done==1, rewards, target_qvalues_for_actions)

        # mean squared error loss to minimize
        loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions) ** 2)

        return loss

    def save(self):
        """Saves model"""        
        with open('Scripts/models/epsilon_greedy_1_model', 'wb') as f:
            pickle.dump(self, f)
        return
