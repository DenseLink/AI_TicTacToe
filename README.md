# Tic-Tac-Toe with AlphaZero

This repository contains the implementation of the Tic-Tac-Toe game and the AlphaZero algorithm, a reinforcement learning approach, to play the game optimally.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Tic-Tac-Toe Game](#tic-tac-toe-game)
4. [AlphaZero Implementation](#alphazero-implementation)
5. [Training](#training)
6. [Testing](#testing)

## Introduction

Tic-Tac-Toe is a simple two-player game played on a 3x3 grid. Players take turns marking X or O in an empty cell. The first player to get three of their marks in a row, column, or diagonal wins. If the entire board is filled without any player winning, the game ends in a draw.

AlphaZero is a deep reinforcement learning algorithm that combines Monte Carlo Tree Search (MCTS) with a neural network. It can learn to play games at a superhuman level without any prior human knowledge of the game.

## Requirements

- Python (>=3.6)
- PyTorch
- NumPy
- tqdm

## Tic-Tac-Toe Game

The TicTacToe class represents the game and provides the following functionalities:

- `get_initial_state()`: Get the initial state of the game, a 3x3 matrix with all cells set to zero.
- `get_next_state(state, action, player)`: Given the current state, the action (index of the cell to mark), and the player (1 or -1), this function returns the new state after the move.
- `get_valid_moves(state)`: Get a binary array indicating valid moves in the current state.
- `check_win(state, action)`: Check if a given action results in a win for the player.
- `get_value_and_terminated(state, action)`: Get the value of the state and if it is a terminal state (win or draw).
- `get_opponent(player)`: Get the opponent of the current player.
- `get_opponent_value(value)`: Get the opponent's value (invert the value).
- `change_perspective(state, player)`: Change the perspective of the board to the given player (1 or -1).
- `get_encoded_state(state)`: Encode the state as a one-hot encoded 3x3x3 matrix.

## AlphaZero Implementation

The AlphaZero algorithm is implemented through several classes:

1. `ResNet`: A convolutional neural network used as the policy and value function approximator in the MCTS. It consists of residual blocks.
2. `ResBlock`: A single residual block used in the ResNet.
3. `Node`: Represents a node in the Monte Carlo Tree Search (MCTS) algorithm.
4. `MCTS`: Implements the Monte Carlo Tree Search algorithm for game exploration.
5. `AlphaZero`: Main class that combines the game, neural network, and MCTS for self-play and training.

## Training

The training process involves self-play and updating the neural network using the collected data from self-play. It includes the following steps:

1. Self-play: The AlphaZero algorithm plays the game against itself using MCTS to explore the game tree and select actions.
2. Data Collection: The states, policy targets (probabilities of actions), and value targets (expected outcomes) are stored during self-play.
3. Neural Network Training: The collected data is used to train the neural network using a combination of policy cross-entropy loss and value mean squared error loss.
4. Iterative Training: The self-play and training process is repeated for a specified number of iterations.

## Testing

The trained AlphaZero model can be tested against a human player. The human player can take turns against the model, and the game outcome will be displayed at the end.

The implementation provided in the code allows for easy experimentation and training of AlphaZero on the Tic-Tac-Toe game. You can modify the parameters and the neural network architecture to suit other games or more complex scenarios.
