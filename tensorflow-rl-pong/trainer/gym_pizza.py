from collections import defaultdict

import numpy as np


L = 1
H = 6


class PizzaGym:
    def __init__(self, best_num_slices=1, min_ingred_in_slice=L, max_cells_in_slice=H):
        self.best_num_slices = best_num_slices
        self.min_ingred_in_slice = min_ingred_in_slice
        self.max_cells_in_slice = max_cells_in_slice
        self.board = None
        self.used_cells = None
        self.num_slices = None
        self.cell_to_ingredient = None

    def reset(self):
        self.board = np.array([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1]
        ])
        self.used_cells = set()
        self.num_slices = 0

        self.cell_to_ingredient = {}
        for r in range(0, self.board.shape[0]):
            for c in range(0, self.board.shape[1]):
                cell = self.get_cell(row=r, col=c)
                self.cell_to_ingredient[cell] = self.board[r][c]

        return self.board

    def step(self, action):
        """
        :param action:
        :return: observ, reward, done, info
        """

        left, right, top, bottom = action

        # check if coordinates are valid

        if left > right or top > bottom:
            return self.return_not_done()

        # infer cells and ingredients

        selected_cells = set()
        selected_ingredients = defaultdict(int)

        for r in range(top, bottom+1):
            for c in range(left, right+1):
                cell = self.get_cell(row=r, col=c)
                selected_cells.add(cell)
                selected_ingredients[self.board[r][c]] += 1

        # check if not selected more than the max allowed cells

        if len(selected_cells) > self.max_cells_in_slice:
            return self.return_not_done()

        # check if all cells were'nt selected in previous slice

        if len(selected_cells.intersection(self.used_cells)) > 0:
            return self.return_not_done()

        # check if min ingredients

        if selected_ingredients[1] < self.min_ingred_in_slice or selected_ingredients[2] > self.min_ingred_in_slice:
            return self.return_not_done()

        # add all cells to selected cells

        for c in selected_cells:
            self.used_cells.add(c)

        # update board

        for r in range(top, bottom+1):
            for c in range(left, right+1):
                self.board[r][c] = 0
                self.cell_to_ingredient[self.get_cell(row=r, col=c)] = 0

        # +1 to number of slices

        self.num_slices += 1

        # check if more slices can be made...

        found_slice = False

        for r in range(0, self.board.shape[0] - 1):
            for c in range(0, self.board.shape[1] - 1):
                cell = self.get_cell(row=r, col=c)
                if cell not in self.used_cells:
                    ingred_1, ingred_2 = self.count_ingredients_in_tree(row=r, col=c, used_cells=self.used_cells)
                    if ingred_1 >= self.min_ingred_in_slice and ingred_2 >= self.min_ingred_in_slice:
                        found_slice = True
                        break

        # return result...

        if found_slice:
            return self.board, 0, False, ''
        else:
            reward = 1 if self.num_slices > self.best_num_slices else -1
            return self.board, reward, True, ''

    def count_ingredients_in_tree(self, row, col, used_cells):
        ingred_1 = 0
        ingred_2 = 0

        cell = self.get_cell(row=row, col=col)
        ingred = self.cell_to_ingredient[cell]
        if ingred == 1:
            ingred_1 = 1
        elif ingred == 2:
            ingred_2 = 1

        used_cells.add(cell)

        if col + 1 < self.board.shape[1]:
            cell_right = self.get_cell(row=row, col=col+1)
            if cell_right not in used_cells:
                i_1, i_2 = self.count_ingredients_in_tree(row=row, col=col+1, used_cells=used_cells)
                ingred_1 += i_1
                ingred_2 += i_2

        if row + 1 < self.board.shape[0]:
            cell_down = self.get_cell(row=row+1, col=col)
            if cell_down not in used_cells:
                i_1, i_2 = self.count_ingredients_in_tree(row=row + 1, col=col, used_cells=used_cells)
                ingred_1 += i_1
                ingred_2 += i_2

        return ingred_1, ingred_2

    def return_not_done(self):
        return self.board, 0, False, ''

    def get_cell(self, row, col):
        return row * self.board.shape[1] + col


def make() -> PizzaGym:
    return PizzaGym()
