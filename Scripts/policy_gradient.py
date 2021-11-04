import torch
import torch.nn as nn
from torch import tensor as tt
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
import pickle


class PolicyGradient(nn.Module):
    """Implementation of Policy Gradient (REINFORCE)"""
    def __init__(self, n_obs, n_actions, lr=1e-3):
        """
        Args:
            n_obs (int): Dimensions of the state space (int for this project)
            n_actions (int): Number of possible actions
            lr (float, optional): Learning rate for the network. Defaults to 1e-3.
        """    
        super().__init__()

        # constants
        self.n_actions = n_actions
        self.n_obs = n_obs

        # networks

        self.model = nn.Sequential(
            nn.Linear(self.n_obs, self.n_obs * 10),  # estimates what to do
            nn.ELU(),
            nn.Linear(self.n_obs * 10, self.n_actions)
        )

    def forward(self, state):
        """Predictions values given state

        Args:
            state (np.array): Observation for given step

        Returns:
            q_values
        """  
        feature = torch.tensor(state, dtype=torch.float32)

        # calculate policy and value function
        policy = self.model(feature)

        return policy

    def get_action(self, state, pcs):
        """ Sample actions with epsilon-greedy policy

        Args:
            state (np.array): Observation for a given step
            pcs (array): Ignore, stands for playable cards but not used

        Returns:
            int: Action to take (card to play)
        """

        policy = self.forward(state)  # use A3C to get policy and value

        # 1. convert policy outputs into probabilities
        # 2. sample the categorical distribution represented by these probabilities
        #policy = policy * torch.tensor(pc)
        action_prob = F.softmax(policy, dim=-1)
        cat = Categorical(action_prob)
        action = cat.sample()

        return int(action), cat.log_prob(action), cat.entropy()


class policy_runner(object):

    def __init__(self, net, env, num_obs=52, num_updates=100000, lr=5e-4):
        """
        Args:
            net (eg_model class): eg_model class, describing a neural network
            env (Trumps env): Custom enviroment made (swap for gym envs for example)
            num_updates (int, optional): Number of games to train for. Defaults to 100000.
            num_obs (int, optional): State space. Defaults to 104.
            lr (float, optional): Learning rate. Defaults to 5e-4.
        """
        super().__init__()

        # constants
        self.entropy_coeff = 0.02
        self.num_obs = net.n_obs
        self.lr = lr

        self.max_grad_norm = 1
        self.num_updates = num_updates
        self.best_loss = np.inf
        self.sum_rewards = []
        self.scores = []

        """Environment"""
        self.env = env

        """Network"""
        self.net = net
        self.opt = torch.optim.Adam(self.net.parameters(), self.lr)

    def train(self):
        """
        Trains model for self.num_updates games
        """
        for episode in range(1, self.num_updates + 1):

            entropy = self.episode_rollout()

            self.opt.zero_grad()

            loss = self.policy_loss(entropy)
            loss.backward()

            self.opt.step()

            if (episode) % 20000 == 0:
                print(f"{episode}")
            if (episode) % 5000 == 0:
                self.evaluate()
            if (episode) % 400 == 0:
                print('.', end='')

            self.sum_rewards.append(self.rewards.sum())

    def evaluate(self):
        """
        Evaluates model performance with epsilon = 0 for 500 games
        """ 
        for _ in range(500):
            total_reward = 0
            s, pc = self.env.reset()
            for __ in range(100):
                a, _, _ = self.net.get_action(s, pc)
                next_s, r, done, pc = self.env.step(a)
                total_reward += r
                s = next_s
                if done:
                    self.scores.append(total_reward)
                    break
        return

    def episode_rollout(self):
        """
        Plays a game and stores all needed features
        """
        episode_entropy = 0
        self.rewards = torch.zeros(5)
        self.values = torch.zeros(5)
        self.log_probs = torch.zeros(5)
        self.actions = torch.zeros(5)
        self.states = torch.zeros((6, self.num_obs))
        self.dones = torch.zeros(5)
        self.state, self.pcs = self.env.reset()

        self.states[0].copy_(torch.from_numpy(self.state))
        for step in range(6):
            """Interact with the environments """
            # call A2C
            a_t, log_p_a_t, entropy = self.net.get_action(self.state, self.pcs)

            # accumulate episode entropy
            episode_entropy += entropy

            # interact
            new_s, reward, done, pcs = self.env.step(a_t)

            self.insert(step, reward, new_s, a_t, log_p_a_t, done)
            self.state = new_s
            self.pcs = pcs

            if done:
                break

        return episode_entropy

    def policy_loss(self, entropy):
        """
        Calculates policy loss

        Args:
            entropy (np.array): represents entropic loss

        Returns:
            [torch.tensor] : Loss function ready to back propagate
        """
        rewards = self._discount_rewards()
        policy_loss = torch.sum((-self.log_probs * rewards), dim=-1)

        loss = policy_loss - self.entropy_coeff * entropy

        return loss

    def _discount_rewards(self, discount=0.99):
        """
        Computes the discounted reward 

        Returns:
            np.array: array of discounted rewards at each timestep
        """        
        r_discounted = torch.zeros(5)

        """Calculate discounted rewards"""
        R = 0
        for i in reversed(range(5)):

            R = self.rewards[i] + discount * R

            r_discounted[i] = R

        return r_discounted

    def insert(self, step, reward, obs, action, log_prob, done):
        """
        Inserts new data into the log for each environment at index step

        Args:
            step (int)): index of the step
            reward (int): reward
            obs (np.array): observation
            action (int): Action taken
            log_prob ([type]): tensor of the log probabilities
            done (int): flag for game is done
        """ 
        self.rewards[step].copy_(torch.from_numpy(np.array(reward)))
        self.states[step + 1].copy_(torch.from_numpy(obs))
        self.actions[step].copy_(torch.tensor(action))
        self.log_probs[step].copy_(log_prob)
        self.dones[step].copy_(torch.from_numpy(np.array(done)))

    def save(self):
        """ Saves model"""
        with open('Scripts/models/policy_gradient_model', 'wb') as f:
            pickle.dump(self, f)
        return
