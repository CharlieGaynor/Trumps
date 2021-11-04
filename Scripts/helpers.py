import numpy as np
import torch
import matplotlib.pyplot as plt


def ma(lst):
    '''Moving average with convolution length 1000'''
    l = 1000
    return np.convolve(lst, np.ones(l), 'valid') / l


def ma2(lst):
    '''Moving average with convolution length 250'''
    l = 250
    return np.convolve(lst, np.ones(l), 'valid') / l


def test_rules_1c(model):
    '''Tests the rules for the 1 card game, epsilon greedy methods
    
    DEPRECATED
    '''
    for card in range(52):
        s = [0 for i in range(52)]
        s[card] = 1
        s = torch.tensor(s, dtype=torch.float32)
        qvals = model.net.forward(s)
        if torch.max(qvals).item() != qvals[card].item():
            return False
    return True


def train_model(model, is_epsilon=False, save=False):
    '''Trains the model and plots the results
    
    Args:
        model (model class): model to tran
        is_epsilon (bool): is the model an epsilon greedy class?
        save (bool): Flag for saving the model down after training
    '''
    if is_epsilon:
        model.train()  # training
        a = ma(model.sum_rewards)  # smoothing curve
        b = ma2(model.scores)

        # plotting
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(range(len(a)), a, linewidth=6)
        ax1.set_title('Training Reward', fontsize=20)
        ax1.set_xlabel('Games played', fontsize=16)
        ax1.set_ylabel('Reward', fontsize=16)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        plt.show()
        plt.clf()

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(range(len(b)), b, color='orange',linewidth=6)  # just one graph at the end
        ax1.set_title('Evaluation reward, ε = 0', fontsize=20)
        ax1.set_xlabel('Evaluations', fontsize=16)
        ax1.set_ylabel('Reward', fontsize=16)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        plt.show()

    else:
        model.train()  # training
        model.evaluate()  # testing with epsilon = 0
        a = ma(model.sum_rewards)  # smoothing curve
        # plotting
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(range(len(a)), a, linewidth=6)
        ax1.set_title('Training Reward', fontsize=20)
        ax1.set_xlabel('Games played', fontsize=16)
        ax1.set_ylabel('Reward', fontsize=16)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
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
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(range(len(a)), a, linewidth=6)
    ax1.set_title('Training Reward', fontsize=20)
    ax1.set_xlabel('Games played', fontsize=16)
    ax1.set_ylabel('Reward', fontsize=16)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    plt.show()
    plt.clf()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(range(len(b)), b, color='orange',linewidth=6)  # just one graph at the end
    ax1.set_title('Evaluation reward, ε = 0', fontsize=20)
    ax1.set_xlabel('Evaluations', fontsize=16)
    ax1.set_ylabel('Reward', fontsize=16)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    plt.show()


def plot_training_graph(sum_rewards):
    '''Helper to plot training graph'''
    a = ma(sum_rewards)  # smoothing curve
    # plotting
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(range(len(a)), a, linewidth=6)
    ax1.set_title('Training Reward', fontsize=20)
    ax1.set_xlabel('Games played', fontsize=16)
    ax1.set_ylabel('Reward', fontsize=16)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    plt.show()
