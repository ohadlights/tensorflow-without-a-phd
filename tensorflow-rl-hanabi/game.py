import numpy as np


MAX_ERRORS = 3


class Game:
    def __init__(self):
        self.done = True
        self.cards_on_table = None
        self.errors = None

    def step(self, action):
        # observ, reward, done, info

        if self.cards_on_table[action] == 0 and self.cards_on_table[action - 1] == 1:
            self.cards_on_table[action] = 1
        else:
            self.errors += 1

        reward = 0

        num_cards_on_table = np.count_nonzero(self.cards_on_table) - 1
        num_errors = self.errors

        if num_cards_on_table == 3 or num_errors == MAX_ERRORS:
            reward = num_cards_on_table - num_errors
            self.done = True

        return self.observation, reward, self.done, {}

    def reset(self):
        self.done = False
        self.errors = 0
        self.cards_on_table = np.zeros(4, np.float32)
        self.cards_on_table[0] = 1
        return self.observation

    @property
    def lowest_reward(self):
        return -MAX_ERRORS

    @property
    def observation(self):
        return self.cards_on_table[1:]

    def render(self):
        print('Table: {}'.format(self.cards_on_table[1:].tolist()))
        print('Error: {}'.format(self.errors))
        print()

