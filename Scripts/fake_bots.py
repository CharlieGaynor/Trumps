# -*- coding: utf-8 -*-
import numpy as np

"""
    List of players invented by me, to try and beat a random opponent (I did).    
"""

def random_player(playable_cards, pile, trump):
    '''input: playable cards, cards on the pile and the trump
    returns a random action'''
    return np.random.choice(playable_cards)


def max_player(playable_cards, pile, trump):
    '''input: playable cards, cards on the pile and the trump
    returns an action based off max cards'''
    cards_mod_13 = np.array(playable_cards) % 13
    return playable_cards[np.argmax(cards_mod_13)]


def min_player(playable_cards, pile, trump):
    '''input: playable cards, cards on the pile and the trump
    returns an action based off min card'''
    cards_mod_13 = np.array(playable_cards) % 13
    return playable_cards[np.argmin(cards_mod_13)]


def great_player(playable_cards, pile, trump):
    """Great player with a little complicated behaviour"""
    cards = playable_cards
    if pile is None:
        return max_player(playable_cards, pile, trump)
    else:
        cards_same_suit = [i for i in cards if i // 13 == pile // 13]
        if len(cards_same_suit) > 0:
            cards_that_will_win = [
                i for i in cards_same_suit if i %
                13 > pile %
                13]
            if len(cards_that_will_win) > 0:
                return min(cards_that_will_win)
            else:
                return min(cards_same_suit)
        else:
            trump_cards = [i for i in cards if i // 13 == trump]
            if len(trump_cards) > 0:
                return min(trump_cards)
            else:
                card_numbers = np.array(cards) % 13
                card_to_play_position = np.argmax(card_numbers)
                return cards[card_to_play_position]


def moving_average(x, w=500):
    """Moving average of length w=500 by default."""
    return np.convolve(x, np.ones(w), 'valid') / w


def simulate(env, player_func=random_player):
    """Simulates games of hardcoded bot vs built-in random AI

    Args:
        env (SuperTrumps class): playground, essentially
        player_func (Function): Strategy for the player. Defaults to random_player.

    Returns:
        list: all the rewards the bot recieved
    """
    total_rewards = []
    for i in range(5000):
        s, pc = env.reset()  # state is cards,pile,trump
        reward_total = 0
        for step in range(102):
            action = player_func(pc, s[1], s[2])
            s, reward, done, pc = env.step(action)
            reward_total += reward

            if done:
                total_rewards.append(reward_total)
                break
    return total_rewards
