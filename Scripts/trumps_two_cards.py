import numpy as np
import random
import torch
import copy
import tensorflow.keras as keras

# We'll start with coding a random AI in. This can be changed to be the own model,
# updated every x time steps. for now just train vs AI
# Do we need a seperate AI to predict?

# I think we could do with a tidier way of passing observations back


# can definitely be sped up eventually, priority: use sort based on suit
# value, trump
def get_get_winner(pile, trump):
    '''
    Gets winner given a pile. Can be made more efficient
    and needs to be
    '''
    if pile[0] // 13 == pile[1] // 13:  # if same suit then choose highest within that suit
        return np.argmax([pile[0] % 13, pile[1] % 13])
    elif pile[1] // 13 == trump:  # if not same suit and second card is a trump then they win
        return 1
    return 0  # else first player wins


def get_playable_cards(cards, rai_card=None):
    '''
    Returns the playable cards for the second player\n
    Takes a list of the players card, and the current suit on the pile
    '''
    if rai_card is None:
        return cards
    suit = rai_card // 13
    x = [c for c in cards if c // 13 == suit]
    if not x:  # If x is empty then return all the cards
        return cards
    return x  # else return allowed cards


def one_hot_playable_cards(cards, rai_card=None):
    if rai_card is None:
        c = get_playable_cards(cards, None)
    else:
        c = get_playable_cards(cards, rai_card)
    return to_one_hot(c)


def get_rai_move(pile, hand):
    '''Gets the move for a random AI.
    This function assumes a 2 player game'''
    if pile == []:
        return random.choice(hand)
    else:
        pc = get_playable_cards(hand, pile[0])
        return random.choice(pc)


def get_reward(round_scores, num_cards=2, predictions=[]):
    '''Returns the reward given round score, number of cards & prediction (optional 4 now)
    '''
    return round_scores / num_cards


def to_one_hot(cards, decksize=52):
    """ helper: take an integer vector and convert it to 1-hot matrix.
    e.g card 3 --> [0,0,0,1,0,0,0...]"""
    y_t = torch.tensor(cards, dtype=torch.int64)
    y_one_hot = torch.zeros(decksize, dtype=torch.int64).scatter(-1, y_t, 1)
    return y_one_hot.numpy()


def get_cards(cards, decksize=52):
    '''
    Input is a list of cards, and the number of cards in the round
    Returns the 'proper' observation for a given number of card. Need to change to 26 cards eventually?
    '''
    return to_one_hot(cards, decksize)  # -1 representing empty)


def get_piles(pile, decksize=52):  # hehe
    '''
    Input is a list of cards in the pile, and the number of players in the round
    Returns the 'proper' observation for a given number of players. May need to set to max players (4?) eventually
    '''
    return to_one_hot(pile, decksize)  # -1 representing empty


def get_trump(trump, length=4):

    return to_one_hot(trump, length)


def get_obs(cards, pile, trump, decksize=52):

    c = to_one_hot(cards, decksize)
    p = to_one_hot(pile, decksize)
    # if len(pile)>0:
    #     p2 = np.array([pile[0]//13])
    #     p3 = np.array([pile[0]%13])
    # else:
    #     #p = np.zeros(52)
    #     p2 = np.array([-1])
    #     p3 = np.array([-1])
    # return np.concatenate([c,p2,p3])
    # if len(pile)>0:
    #     p = np.array([pile[0]%13,pile[0]//13])
    # else:
    #     p = np.array([-5,-5])

    combined = np.concatenate([c, p])
    return combined


class SuperTrumps:

    def __init__(self, num_cards=2, decksize=52):
        deck = np.arange(0, decksize, 1)
        np.random.shuffle(deck)
        self.deck_size = decksize
        self.num_cards = num_cards
        self.max_cards = decksize
        self.raicards = deck[0:self.num_cards]
        self.cards = np.array(deck[self.num_cards:self.num_cards * 2])
        self.trump = (deck[self.num_cards * 2]) // 13  # (int 0,1,2,3)
        self.pile = []  # this needs to depend on who playing first
        self.done = 0
        self.rounds_won = 0
        # (max_cards+num_players+1,) #cards in hand, pile, trump, not sure if needed
        self.observation_space = (
            len(get_obs([i for i in range(num_cards)], [0], 4)),)
        self.action_space = decksize
        self.player_first = -1

    def step(self, action):
        '''Computes a step in the game
        action is an integer representing a card, 1-52
        assume for now it must be in the hand \n
        returns observation (why), reward, is done'''

        if self.pile == []:
            if action in self.cards:  # All cards are playable here
                self.cards = np.setdiff1d(
                    self.cards, action)  # Remove card from pile
                self.pile.append(action)  # Put it on the pile
                #### Compute AI's movement ####
                rai_action = get_rai_move(self.pile, self.raicards)
                self.raicards = np.setdiff1d(self.raicards, rai_action)
                self.pile.append(rai_action)
                ##### Getting winner #####
                winner = get_get_winner(self.pile, self.trump)
                self.pile = []
                ###Computing next step, depending who the winner is###
                if winner == 0:  # If our bot won
                    self.rounds_won += 1
                    # might need to specifically point out which cards are
                    # trumps to aid training?
                    new_obs = get_obs(self.cards, self.pile, self.trump)

                    if len(self.cards) > 0:
                        # +1 reward for now as we want to maximise rounds won, do these need unpacking?
                        return (
                            new_obs,
                            1,
                            self.done,
                            one_hot_playable_cards(
                                self.cards))
                    else:
                        self.done = 1
                        # Reward needs to be calculated but for now just +1
                        return (
                            new_obs,
                            1,
                            self.done,
                            one_hot_playable_cards(
                                self.cards))

                elif winner == 1:  # If the RAI wonreturn

                    if len(self.raicards) == 0:
                        self.done = 1
                        new_obs = get_obs(self.cards, self.pile, self.trump)
                        return (
                            new_obs,
                            0,
                            self.done,
                            one_hot_playable_cards(
                                self.cards))
                    else:
                        rai_action = get_rai_move(self.pile, self.raicards)
                        self.raicards = np.setdiff1d(self.raicards, rai_action)
                        # This is the number of which card is there (I think)
                        self.pile = [rai_action]
                        new_obs = get_obs(self.cards, self.pile, self.trump)
                        return (
                            new_obs, 0, self.done, one_hot_playable_cards(
                                self.cards, rai_action))
            else:
                self.done = 1
                new_obs = get_obs(self.cards, self.pile, self.trump)
                return (new_obs, -10, self.done,
                        one_hot_playable_cards(self.cards))

        elif self.pile != []:  # if pile is not empty
            playable_cards = get_playable_cards(self.cards, (self.pile[0]))
            if action in playable_cards:  # All cards are playable here
                self.cards = np.setdiff1d(
                    self.cards, action)  # Remove card from pile
                self.pile.append(action)  # Put it on the pile
                ##### Getting winner #####
                winner = get_get_winner(self.pile, self.trump)
                self.pile = []
                ###Computing next step, depending who the winner is###
                if winner == 1:  # If our bot won
                    self.rounds_won += 1
                    new_obs = get_obs(self.cards, self.pile, self.trump)
                    if len(self.cards) > 0:
                        return (
                            new_obs, 1, self.done, one_hot_playable_cards(
                                self.cards))  # need to add observation
                    else:
                        self.done = 1
                        # Reward needs to be calculated, obs needs to be filled
                        # in
                        return (
                            new_obs,
                            1,
                            self.done,
                            one_hot_playable_cards(
                                self.cards))

                elif winner == 0:  # If the RAI wonreturn
                    if len(self.cards) == 0:
                        self.done = 1
                        new_obs = get_obs(self.cards, self.pile, self.trump)
                        return (
                            new_obs,
                            0,
                            self.done,
                            one_hot_playable_cards(
                                self.cards))
                    else:
                        rai_action = get_rai_move(self.pile, self.raicards)
                        self.raicards = np.setdiff1d(self.raicards, rai_action)
                        self.pile = [rai_action]
                        new_obs = get_obs(self.cards, self.pile, self.trump)
                        return (
                            new_obs, 0, self.done, one_hot_playable_cards(
                                self.cards, rai_action))
            else:
                self.done = 1
                new_obs = get_obs(self.cards, self.pile, self.trump)
                return (new_obs, -10, self.done,
                        one_hot_playable_cards(self.cards))

    def reset(self):
        '''
        Resets the game ready for another round, should return an initial observation only
        '''
        self.player_first = 1
        deck = np.arange(0, self.deck_size, 1)  # Cards 1-51, no jokers for now
        np.random.shuffle(deck)  # Shuffles
        self.raicards = deck[0:self.num_cards]  # random_AI cards
        self.cards = deck[self.num_cards:self.num_cards * 2]  # ai cards
        self.trump = deck[self.num_cards * 2] // 13
        self.pile = []
        self.done = 0
        self.rounds_won = 0
        return (
            get_obs(
                self.cards,
                self.pile,
                self.trump),
            one_hot_playable_cards(
                self.cards))
        # need something to choose player for next round, for now always just
        # A.I first

    def print_info(self):
        '''
        Prints info about current state of game
        '''
        print('Our Cards: ', self.cards,
              '\nRAI cards: ', self.raicards,
              '\nPile: ', self.pile)
        if self.done == 1:
            print('Reward:', get_reward(self.rounds_won))

        return
