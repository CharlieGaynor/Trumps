import numpy as np
import torch
import matplotlib.pyplot as plt


def ma(lst):
    l = 1000
    return np.convolve(lst, np.ones(l), 'valid') / l


def ma2(lst):
    l = 250
    return np.convolve(lst, np.ones(l), 'valid') / l


def test_rules_1c(model):
    for card in range(52):
        s = [0 for i in range(52)]
        s[card] = 1
        s = torch.tensor(s, dtype=torch.float32)
        qvals = model.net.forward(s)
        if torch.max(qvals).item() != qvals[card].item():
            return False
    return True


def train_model(model, is_epsilon=False, save=False):

    if is_epsilon:
        model.train()  # training
        a = ma(model.sum_rewards)  # smoothing curve
        b = ma2(model.scores)

        # plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(range(len(a)), a)
        ax1.set_title('Training Reward')
        ax1.set_xlabel('Games played')
        ax1.set_ylabel('Reward')
        ax2.plot(range(len(b)), b, color='orange')  # just one graph at the end
        ax2.set_title('Evaluation reward, ε = 0')
        ax2.set_xlabel('Evaluations')
        ax2.set_ylabel('Reward')
        fig.tight_layout()
        plt.show()

    else:
        model.train()  # training
        model.evaluate()  # testing with epsilon = 0
        a = ma(model.sum_rewards)  # smoothing curve
        # plotting
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(range(len(a)), a)
        ax1.set_title('Training Reward')
        ax1.set_xlabel('Games played')
        ax1.set_ylabel('Reward')
        plt.show()

    print('Is the model still making mistakes?')
    for _ in range(4):
        model.evaluate()
    mistakes = False if min(model.scores[-2000:]) >= 0 else True
    print(mistakes)

    if save:
        print('saving model...')
        model.save()


def plot_both_graphs(sum_rewards, scores):
    '''Helper to plot training and evaluation information
    for a loaded model'''
    # First smothing curves
    a = ma(sum_rewards)
    b = ma2(scores)
    # Now Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(range(len(a)), a)
    ax1.set_title('Training Reward')
    ax1.set_xlabel('Games played')
    ax1.set_ylabel('Reward')
    ax2.plot(range(len(b)), b, color='orange')  # just one graph at the end
    ax2.set_title('Evaluation reward, ε = 0')
    ax2.set_xlabel('Evaluations')
    ax2.set_ylabel('Reward')
    fig.tight_layout()
    plt.show()


def plot_training_graph(sum_rewards):
    '''Helper to plot training graph'''
    a = ma(sum_rewards)  # smoothing curve
    # plotting
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(range(len(a)), a)
    ax1.set_title('Training Reward')
    ax1.set_xlabel('Games played')
    ax1.set_ylabel('Reward')
    plt.show()
