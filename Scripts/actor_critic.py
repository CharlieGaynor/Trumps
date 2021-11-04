import torch
import torch.nn as nn
from torch import tensor as tt
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
import pickle

class A2Cnet(nn.Module):
    """Implementation of the Advantage Actor-Critic (A2C) network"""
    def __init__(self, n_obs, n_actions,lr=1e-3):
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

        self.actor = nn.Sequential(
            nn.Linear(self.n_obs, self.n_obs *10),  # Gives values for each action
            nn.ELU(),
            nn.Linear(self.n_obs*10,self.n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.n_obs, self.n_obs*10), 
            nn.ELU(),
            nn.Linear(self.n_obs*10,1)
        )  # estimates how good the value function (how good the current state is)

    def forward(self, state):
        """Predictions values given state

        Args:
            state (np.array): Observation for given step

        Returns:
            q_values
        """  
        feature = torch.tensor(state,dtype=torch.float32)

        # calculate policy and value function
        policy = self.actor(feature)
        value = self.critic(feature)

        return policy, torch.squeeze(value)

    def get_action(self, state,pcs):
        """ Sample actions with epsilon-greedy policy

        Args:
            state (np.array): Observation for a given step
            pcs (array): Ignore, stands for playable cards but not used

        Returns:
            int: Action to take (card to play)
        """

        """Evaluate the A2C"""
        policy, value = self(state)  # use A3C to get policy and value

        """Calculate action"""
        # 1. convert policy outputs into probabilities
        # 2. sample the categorical  distribution represented by these probabilities
        #policy = policy * torch.tensor(pc)
        action_prob = F.softmax(policy, dim=-1)
        cat = Categorical(action_prob)
        action = cat.sample()

        return (action, cat.log_prob(action), cat.entropy().mean(), value)


class ac_runner(object):

    def __init__(self, net, env,num_updates=100000,num_obs =104 ,lr=5e-4,is_cuda=False):
        """
        Args:
            net (eg_model class): eg_model class, describing a neural network
            env (Trumps env): Custom enviroment made (swap for gym envs for example)
            num_updates (int, optional): Number of games to train for. Defaults to 100000.
            num_obs (int, optional): State space. Defaults to 104.
            lr (float, optional): Learning rate. Defaults to 5e-4.
            is_cuda (keep false plz)
        """
        super().__init__()

        # constants
        self.value_coeff = 0.5
        self.entropy_coeff = 0.002
        self.num_obs = num_obs
        self.lr = lr

        self.max_grad_norm = 1
        self.num_updates= num_updates
        self.best_loss = np.inf
        self.sum_rewards = []
        self.scores = []


        # Whether to use a GPU, ignore for now
        self.is_cuda = torch.cuda.is_available() and is_cuda


        """Environment"""
        self.env = env

        """Network"""
        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr)

        if self.is_cuda:
            self.net = self.net.cuda()

    def train(self):
        """
        Trains model for self.num_updates games
        """

        """Environment reset"""
        self.state,self.pcs = self.env.reset()

        for episode in range(1,self.num_updates+1):

            final_value, entropy = self.episode_rollout()

            self.optimizer.zero_grad()


            """Assemble loss"""
            loss = self.a2c_loss(final_value, entropy)
            loss.backward(retain_graph=False)
            
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

            self.optimizer.step()

            # Printing stuff
            if (episode) % 20000 == 0:
                print(f"{episode}")
            if (episode) % 5000 == 0:
                self.evaluate()
            if (episode) % 400 == 0:
                print('.',end='')
            
            # Storing for later
            self.sum_rewards.append(self.rewards.sum())

    def episode_rollout(self):
        """
        Plays a game and stores all needed features
        """
        episode_entropy = 0
        self.rewards = torch.zeros(5)
        self.values = torch.zeros(5)
        self.log_probs = torch.zeros(5)
        self.actions = torch.zeros(5)
        self.states = torch.zeros((6,self.num_obs))
        self.dones = torch.zeros(5)
        self.state, self.pcs = self.env.reset()
        
        self.states[0].copy_(torch.from_numpy(self.state))
        for step in range(6):
            """Interact with the environments """
            # call A2C
            a_t, log_p_a_t, entropy, value = self.net.get_action(self.state,self.pcs)
            # accumulate episode entropy, no longer needed but scared to break stuff!
            episode_entropy += entropy

            # Take a step
            new_s, reward, done, pcs = self.env.step(int(a_t.cpu()))

            self.insert(step,reward,new_s,a_t,log_p_a_t,value,done)
            self.state = new_s
            self.pcs = pcs

            if done:
                break

        with torch.no_grad():
            _, _, _, final_value = self.net.get_action(self.state,self.pcs)

        return final_value, episode_entropy

    def evaluate(self):
        """
        Evaluates model performance with epsilon = 0 for 500 games
        """ 
        for blank in range(500):
            total_reward = 0
            s,pc = self.env.reset()
            for __ in range(100):
                a_t, _,_,_ = self.net.get_action(s,pc)

                # Take a step
                new_s, reward, done, pc = self.env.step(int(a_t.cpu()))
                total_reward += reward
                s = new_s

                if done:
                    self.scores.append(total_reward)
                    break
        return
    
    def a2c_loss(self, final_value, entropy):
        """
        Calculates a2c loss

        Args:
            entropy (np.array): represents entropic loss

        Returns:
            [torch.tensor] : Loss function ready to back propagate
        """
        rewards = self._discount_rewards(final_value)
        advantage = rewards - self.values
        policy_loss = (-self.log_probs * advantage.detach()).sum()

        value_loss = advantage.pow(2).sum()

        loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy


        return loss

    def _discount_rewards(self, final_value, discount=0.99):
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
    
    def insert(self, step, reward, obs, action, log_prob, value, done):
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
        self.actions[step].copy_(action)
        self.log_probs[step].copy_(log_prob)
        self.values[step].copy_(value)
        self.dones[step].copy_(torch.from_numpy(np.array(done)))

    def save(self):
        """ Saves model"""
        with open('Scripts/models/actor_critic_model','wb') as f:
            pickle.dump(self,f)
        return
        

