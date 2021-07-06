# ConvolutedSudoku
A new variant of sudoku using 2D convolution

The main.py file is a tool for setting and checking convoluted sudoku. It is currently very simplistic, and only recognizes the convolutional constraint and n-tuples in rows, columns, and boxes. 

Currently it is hardcoded to use the convolution matrix
||||
|---|---|---|
|  0 | -1 |  0 |
| -1 |  4 | -1 |
|  0 | -1 |  0 |

so when a convolution total is given, that number is equal to 4 times the digit in that cell, minus each orthogonally connected digit.

For example, the convolution total 29 can only be created by having the digit 9 in the center with two 1s, one 2, and one 3 orthogonally connected.
