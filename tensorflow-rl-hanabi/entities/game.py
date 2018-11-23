import numpy as np

from entities.color import *


class Game:

    CARD_ACTIONS = {
        0: (1, BLUE),
        1: (2, BLUE),
        2: (3, BLUE),
        3: (1, GREEN),
        4: (2, GREEN),
        5: (3, GREEN),
    }

    NUM_ACTIONS = len(CARD_ACTIONS)

    COLORS = set([a[1] for a in CARD_ACTIONS.values()])

    CARDS_MAX_VALUE = max(a[0] for a in CARD_ACTIONS.values())

    MAX_ERRORS = 3

    def __init__(self):
        self.done = True
        self.cards_on_table = None
        self.errors = None

    def reset(self):
        self.done = False
        self.errors = 0
        self.cards_on_table = np.zeros((len(Game.COLORS), 4), np.float32)
        self.cards_on_table[:, 0] = 1
        return self.observation

    def step(self, action):
        # observ, reward, done, info

        card_action = Game.CARD_ACTIONS[action]
        card_index = card_action[0]
        card_color = card_action[1]

        cards_on_table = self.cards_on_table[card_color.color_id]

        if cards_on_table[card_index] == 0 and cards_on_table[card_index - 1] == 1:
            cards_on_table[card_index] = 1
        else:
            self.errors += 1

        reward = 0

        num_cards_on_table = np.count_nonzero(self.observation)
        num_errors = self.errors

        if num_cards_on_table == len(Game.CARD_ACTIONS) or num_errors == Game.MAX_ERRORS:
            ratio = int(len(Game.CARD_ACTIONS) / Game.MAX_ERRORS)
            reward = num_cards_on_table - (num_errors * ratio)
            self.done = True

        return self.observation, reward, self.done, {}

    @property
    def lowest_reward(self):
        return -Game.MAX_ERRORS

    @property
    def observation(self):
        return self.cards_on_table[:, 1:]

    def render(self):
        print('Table: {}'.format(self.observation.tolist()))
        print('Error: {}'.format(self.errors))
        print()

