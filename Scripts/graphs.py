from architecture import *
from icecream import ic
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def get_reward(method, type, ax, n_cards):

    x = SuperTrumps(n_cards)
    rewards = []
    for _ in range(1000):
        done = False
        while not done:
            cards = [(c - 1) % 13 for c in x.cards]
            if type == 1:
                index = cards.index(max(cards))
            elif type == 2:
                index = cards.index(min(cards))
            elif type == 3:
                index = 0
            elif type == 4:
                trump = x.trump
                trump_cards = [
                    (c - 1) %
                    13 if (
                        c - 1) // 13 == trump else 0 for c in x.cards]
                if sum(trump_cards) != 0:
                    index = trump_cards.index(max(trump_cards))
                else:
                    index = cards.index(max(cards))

            obs, reward, done = x.step(x.cards[index])
        rewards.append(reward)
        x = SuperTrumps(n_cards)

    df = pd.DataFrame(data={f'reward_{method}': rewards})
    cumsum = df[f'reward_{method}'].cumsum()
    cumavg = [cumsum[i - 1] / i for i in range(1, len(cumsum) + 1)]

    # plt.hist(df.reward)

    ax.scatter(range(len(cumavg)), cumavg, label=f'{method}')


fig, axes = plt.subplots(2, 2)

for j, num_cards in enumerate([2, 5, 10, 20]):

    ax = axes[j // 2][j % 2]
    n_cards = num_cards

    get_reward('max', 1, ax, n_cards)
    get_reward('min', 2, ax, n_cards)
    get_reward('random', 3, ax, n_cards)
    # get_reward('trump_first',4,ax,n_cards)
    ax.legend()
    ax.set_title(f'Number of cards = {n_cards}')

fig.tight_layout()
plt.show()
