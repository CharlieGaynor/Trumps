import numpy as np
import random
from icecream import ic
'''
Ye got ye old cards:0-51
4 suits, how to assign?

try 2 people for now, but also try to keep in general ish

create a deck, be able to shuffle, be able to deal
draw the cards in console

play: (check if valid move)
who moves next


evaluate who wins:
if any card has self.suit == trump then highest
'''
N = 2


class hand:  # short for hand
    def __init__(self, cards):
        '''
        cards will be a list of card objects
        '''
        self.cards = cards

    def show_hand(self):
        '''
        print hand for user (only useful for humans)
        '''
        print(self.cards)

    def play_card(self, given_card):
        '''
        card will be a tuple of (suit,value)
        '''
        try:
            self.cards.remove(given_card)
        except ValueError:
            print('You do not have that card fool')  # for human game only

        '''need the next step to play'''

    def playable_cards(self, suit):
        x = [c for c in self.cards if c[0] == suit]
        if not x:
            return self.cards.copy()
        return x


class game:  # short for game, obviously

    def __init__(self, number_of_cards=4):
        self.number_of_cards = number_of_cards
        self.start_player = 0

    def deal_hands(self, N):
        x = []
        deck = [(j, i) for j in range(4) for i in range(2, 15)]
        np.random.shuffle(deck)
        for i in range(N):
            x.append(hand(deck[self.number_of_cards *
                     i:self.number_of_cards * (i + 1)]))

        trump = deck[self.number_of_cards * N]
        return x, trump

    def play_round(self):
        '''
        Execute one 'round' , which a single card played by each player
        '''
        pass


class player:

    def __init__(self, type, round_score=0):
        self.type = type  # 0 for human, , 1 for random, 2 for bot
        # self.score = score worry about this later
        self.value = 0
        self.round_score = round_score

    def play_move(self, hand, suit=False):

        if suit:
            pc = hand.playable_cards(suit).copy()  # playable cards
        else:
            pc = hand.cards.copy()

        if self.type == 0:
            hand.show_hand()  # could put in while loop - u nobhead nils
            while True:
                try:
                    # user input
                    ui = input('select (suit,value) of card to play')
                    c_toplay = tuple(int(x) for x in ui.split(","))
                    if c_toplay in pc:
                        hand.cards.remove(c_toplay)
                        return c_toplay
                except ValueError:
                    pass
                print('You floopser')

        if self.type == 1:
            c_toplay = pc[random.randint(0, len(pc) - 1)]
            hand.cards.remove(c_toplay)
            print(c_toplay)
            return c_toplay

# or pick highest trump, or highest suit


# can definitely be sped up eventually, priority: use sort based on suit
# value, trump
def get_get_winner(pile, suit, trump):
    temp_winner = 0
    for i in range(1, len(pile)):
        if pile[temp_winner][0] == pile[i][0]:
            temp_winner = np.where(
                pile[i][1] > pile[temp_winner][1], i, temp_winner)
        elif pile[i][0] == trump:
            temp_winner = i
    return temp_winner


num_of_cards = 4
curr_game = game(num_of_cards)
chunky_player = 1
players = []
types = [0, 1]  # needs to be specified
for j in range(N):
    players.append(player(types[j]))

hands, trump = curr_game.deal_hands(N)
curr_player = chunky_player
for round in range(num_of_cards):

    print(f'trump = {trump}')
    card_pile = [players[curr_player].play_move(hands[curr_player])]
    suit = card_pile[0][0]

    for i in range(curr_player + 1, N + curr_player):
        i = i % N
        card_pile.append(players[i].play_move(hands[i], suit))

    winner = get_get_winner(card_pile, suit, trump)
    curr_player = (curr_player + winner) % N

    players[curr_player].round_score += 1

    # whoever wins, curr_player = px.value
for playa in players:
    print(playa.round_score, end=' ')

chunky_player = (chunky_player + 1) % 2  # mod number of players


#start_player += 1 %2 #### mod number of players, not always 2 ####

'''

next time:

- chunks, keep track fo score (+1 if won round, 0 if didnt, 0.5 if draw)
- execute a full game (multiple chunks)
- change suit to S,H,C,D
- make a very very simple ai (alwys pick largest card possible)

'''

''''''
