import copy
import functools
from collections.abc import Iterable
from typing import Optional, List, Set, Tuple
import multiprocessing

import tqdm
from joblib import Parallel, delayed

VERBOSE = True
SETTING = False

sudoku_possibilities = [[set(range(1, 10)) for col in range(9)] for row in range(9)]
sudoku_convolution = [
    [  18, None, None,   None, None, None,   None, None, None],
    [None, None, None,     11, None, None,   None, None, None],
    [None, None,    0,   None, None, None,   None, None, None],

    [None,    0, None,     29, None,    0,   None, None, None],
    [None, None,   -4,   None,    0, None,    -27, None, None],
    [None,    2, None,    -10, None,  -29,   None, None, None],

    [None, None,    0,   None,  -28, None,   None, None, None],
    [None, None, None,   None, None, None,   None, None, None],
    [ -12, None, None,   None, None, None,   None, None,   18],
]
sudoku_definite = [
    [None, None, None,   None, None, None,   None, None, None],
    [None, None, None,   None, None, None,   None, None, None],
    [None, None, None,   None, None, None,   None, None, None],

    [None, None, None,   None, None, None,   None, None, None],
    [None, None, None,   None, None, None,   None, None, None],
    [None, None, None,   None, None, None,   None, None, None],

    [None, None, None,   None, None, None,   None, None, None],
    [None, None, None,   None, None, None,   None, None, None],
    [None, None, None,   None, None, None,   None, None, None],
]


class ConvolutedSudokuCell:
    def __init__(self, row: int, col: int, conv: Optional[int] = None, given: Optional[int] = None):
        self.row = row
        self.col = col
        self.possibilities: Set[int] = set(range(1, 10))
        self.conv = conv
        self.value = given
        self.up: Optional['ConvolutedSudokuCell'] = None
        self.down: Optional['ConvolutedSudokuCell'] = None
        self.left: Optional['ConvolutedSudokuCell'] = None
        self.right: Optional['ConvolutedSudokuCell'] = None
        self.parent: Optional['ConvolutedSudoku'] = None

    def set_convolution(self, value: int):
        total_possibilities = []
        left_possibilities: Set[int] = set()
        right_possibilities: Set[int] = set()
        top_possibilities: Set[int] = set()
        bottom_possibilities: Set[int] = set()
        self.conv = value
        for c in self.possibilities:
            left_possibilities = self.left.possibilities.copy() if self.left is not None else {0}
            left_possibilities.discard(c)
            for L in left_possibilities:
                right_possibilities = self.right.possibilities.copy() if self.right is not None else {0}
                right_possibilities.discard(c)
                if L > 0:
                    right_possibilities.discard(L)
                for r in right_possibilities:
                    top_possibilities = self.up.possibilities.copy() if self.up is not None else {0}
                    top_possibilities.discard(c)
                    if self.row % 3 > 0:
                        if self.col % 3 < 2 and r > 0:
                            top_possibilities.discard(r)
                        if self.col % 3 > 0 and L > 0:
                            top_possibilities.discard(L)
                    for t in top_possibilities:
                        bottom_possibilities = self.down.possibilities.copy() if self.down is not None else {0}
                        bottom_possibilities.discard(c)
                        if t > 0:
                            bottom_possibilities.discard(t)
                        if self.row % 3 < 2:
                            if self.col % 3 < 2 and r > 0:
                                bottom_possibilities.discard(r)
                            if self.col % 3 > 0 and L > 0:
                                bottom_possibilities.discard(L)
                        for b in bottom_possibilities:
                            total_possibilities.append((c, L, r, t, b))
        conv_possibilities: Set[Tuple[int, int, int, int, int]] = set()
        for c, L, r, t, b in total_possibilities:
            if 4*c - (L + r + t + b) == self.conv:
                conv_possibilities.add((c, L, r, t, b))
        center_possibilities: Set[int] = set()
        left_possibilities.clear()
        right_possibilities.clear()
        top_possibilities.clear()
        bottom_possibilities.clear()

        for c, L, r, t, b in conv_possibilities:
            center_possibilities.add(c)
            left_possibilities.add(L)
            right_possibilities.add(r)
            top_possibilities.add(t)
            bottom_possibilities.add(b)
        old_changed = self.parent.changed
        self.parent.changed = False
        self.set_possibilities(center_possibilities)
        if self.left is not None:
            self.left.set_possibilities(left_possibilities)
        if self.right is not None:
            self.right.set_possibilities(right_possibilities)
        if self.up is not None:
            self.up.set_possibilities(top_possibilities)
        if self.down is not None:
            self.down.set_possibilities(bottom_possibilities)
        if VERBOSE and self.parent.changed:
            self.parent.print()
            print(f"Applied convolution constraint to row {self.row + 1}, column {self.col + 1}.")
            input("Press enter to continue.")
        self.parent.changed = self.parent.changed or old_changed

    def set_possibilities(self, values: Set[int]):
        new_possibilities = self.possibilities.intersection(values)
        if new_possibilities != self.possibilities:
            self.parent.changed = True
        self.possibilities = new_possibilities
        if len(self.possibilities) == 0:
            raise ValueError("You made the puzzle impossible, dummy!")
        elif len(self.possibilities) == 1:
            self.set_definite(next(iter(self.possibilities)))

    def set_definite(self, value: int):
        if self.value is not None:
            if self.value == value:
                return
            else:
                raise ValueError("You broke the puzzle, dummy!")
        if VERBOSE:
            self.parent.print()
            print(f"Determined row {self.row + 1}, column {self.col + 1} to be {value}.")
            input("Press enter to continue.")
        self.value = value
        self.parent.changed = True
        for r in range(9):
            if r == self.row:
                continue
            self.parent.grid[r][self.col].remove_possibilities({value})
        for c in range(9):
            if c == self.col:
                continue
            self.parent.grid[self.row][c].remove_possibilities({value})
        box_x = self.col // 3
        box_y = self.row // 3
        for r in range(box_y*3, (box_y+1)*3):
            for c in range(box_x*3, (box_x+1)*3):
                if r == self.row and c == self.col:
                    continue
                self.parent.grid[r][c].remove_possibilities({value})

    def remove_possibilities(self, values: Set[int]):
        new_possibilities = self.possibilities.difference(values)
        if new_possibilities != self.possibilities:
            self.parent.changed = True
        self.possibilities = new_possibilities
        if len(self.possibilities) == 0:
            raise ValueError("You made the puzzle impossible, dummy!")
        elif len(self.possibilities) == 1:
            self.set_definite(next(iter(self.possibilities)))


class ConvolutedSudoku:
    def __init__(self):
        self.grid = [[ConvolutedSudokuCell(r, c) for c in range(9)] for r in range(9)]
        self.changed = False
        for r in range(9):
            for c in range(9):
                cell = self.grid[r][c]
                cell.parent = self
                if c > 0:
                    cell.left = self.grid[r][c-1]
                if c < 8:
                    cell.right = self.grid[r][c+1]
                if r > 0:
                    cell.up = self.grid[r-1][c]
                if r < 8:
                    cell.down = self.grid[r+1][c]

    def print(self):
        head = "╔═════╤═════╤═════╦═════╤═════╤═════╦═════╤═════╤═════╗"
        foot = "╚═════╧═════╧═════╩═════╧═════╧═════╩═════╧═════╧═════╝"
        row_h = ["║", "│", "│"]*3
        print(head)
        for r in range(9):
            for line in range(3):
                for c in range(9):
                    cell = self.grid[r][c]
                    print(row_h[c], end="")
                    if cell.value is not None:
                        if line == 0:
                            print("▛▀▀▀▜", end="")
                        elif line == 1:
                            print(f"▌ {cell.value} ▐", end="")
                        else:
                            print("▙▄▄▄▟", end="")
                    else:
                        print(" ", end="")
                        for p in range(3*line+1,3*line+4):
                            if p in cell.possibilities:
                                print(p, end="")
                            else:
                                print(" ", end="")
                        print(" ", end="")
                print(row_h[0])
            if r == 8:
                print(foot)
            elif r % 3 == 2:
                s = "╠" + "╬".join(["╪".join(["═"*5]*3)]*3) + "╣"
                print(s)
            else:
                s = "╟" + "╫".join(["┼".join(["─" * 5] * 3)] * 3) + "╢"
                print(s)

    def set_convolution(self, convolution: List[List[Optional[int]]]):
        for r in range(9):
            for c in range(9):
                if convolution[r][c] is not None:
                    self.grid[r][c].set_convolution(convolution[r][c])

    def check_tuples(self):
        def choose_items(n: int, vals: Iterable):
            if n < 1:
                raise ValueError("Can't choose fewer than 1 item!")
            if n == 1:
                for val in vals:
                    yield {val}
            else:
                already_done = []
                for subchoose in choose_items(n-1, vals):
                    if len(subchoose) != n-1:
                        raise ValueError("I don't know how, but you broke it.")
                    for val in vals:
                        if val in subchoose or {val, *subchoose} in already_done:
                            continue
                        else:
                            already_done.append({val, *subchoose})
                            yield {val, *subchoose}

        def check_tuples_on_cells(all_cells: Iterable[ConvolutedSudokuCell], name: Optional[str] = None):
            for num in range(2, 9):
                for cell_set in choose_items(num, all_cells):
                    p_union = functools.reduce(lambda a, b: a.union(b), [cell.possibilities for cell in cell_set])
                    if len(p_union) < num:
                        raise ValueError("You broke the puzzle, dummy!")
                    elif len(p_union) == num:
                        old_changed = self.changed
                        self.changed = False
                        for cell in all_cells:
                            if cell not in cell_set:
                                cell.remove_possibilities(p_union)
                        if VERBOSE and self.changed:
                            if name is None:
                                name = "unnamed cell set"
                            self.print()
                            print(f"Found {', '.join([str(x) for x in sorted(p_union)])} tuple in {name}.")
                            input("Press enter to continue.")
                        self.changed = self.changed or old_changed
        for r in range(9):
            check_tuples_on_cells(self.grid[r], name=f"row {r+1}")
        for c in range(9):
            check_tuples_on_cells([self.grid[r][c] for r in range(9)], name=f"column {c+1}")
        for box_x in range(3):
            for box_y in range(3):
                check_tuples_on_cells([self.grid[r][c] for c in range(box_x*3, (box_x+1)*3) for r in range(box_y*3, (box_y+1)*3)], name=f"box {box_y*3+box_x+1}")


def main():
    def run_until_no_change(cs: ConvolutedSudoku, cv: List[List[Optional[int]]]):
        cs.set_convolution(cv)
        cs.check_tuples()
        while cs.changed:
            cs.changed = False
            cs.set_convolution(cv)
            cs.check_tuples()

    sudoku = ConvolutedSudoku()
    run_until_no_change(sudoku, sudoku_convolution)
    while SETTING:
        sudoku.print()
        row = int(input("Which row would you like to investigate? "))
        col = int(input(f"Which column of row {row} would you like to investigate? "))

        def test_value(n: int):
            test_convolution = copy.deepcopy(sudoku_convolution)
            test_convolution[row][col] = n
            try:
                test_sudoku = ConvolutedSudoku()
                run_until_no_change(test_sudoku, test_convolution)
                return True
            except ValueError:
                return False
        inputs = tqdm.tqdm(range(-29, 30))
        allowed = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(test_value)(n) for n in inputs)
        allowed_values = [value for value, allow in zip(range(-29, 30), allowed) if allow]
        val = input(f"The allowed values for cell ({row}, {col}) are {allowed_values}. Which of these would you like to use? ")
        try:
            sudoku_convolution[row][col] = int(val)
            print(f"Using {int(val)} for cell ({row}, {col})...")
        except ValueError:
            continue
        sudoku = ConvolutedSudoku()
        run_until_no_change(sudoku, sudoku_convolution)



if __name__ == "__main__":
    main()
