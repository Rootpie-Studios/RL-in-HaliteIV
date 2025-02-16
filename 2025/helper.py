from kaggle_environments.envs.halite.helpers import *
import numpy as np
import math


def ship_action_index(action_string):
    """Takes input action string and returns number representation"""

    if action_string == "EAST":
        return 1
    elif action_string == "NORTH":
        return 2
    elif action_string == "SOUTH":
        return 3
    elif action_string == "WEST":
        return 4
    else:  # CONVERT
        return 5


def ship_action_string(action_index):
    """Takes input action string and returns number representation"""

    if action_index == 1:
        return ShipAction.EAST
    elif action_index == 2:
        return ShipAction.NORTH
    elif action_index == 3:
        return ShipAction.SOUTH
    elif action_index == 4:
        return ShipAction.WEST
    elif action_index == 5:
        return ShipAction.CONVERT


def center_cell(input_list, center_pos):
    """Takes input_list (=one image representation of data from game 21x21) and returns it as
    an equivalent representation with input center_pos as center cell"""

    size = int(math.sqrt(len(input_list)))
    half_size = math.floor(size / 2)

    mat = np.reshape(input_list, (size, size))
    res_mat = np.zeros([size * 2, size * 2])

    res_mat[0 : 0 + mat.shape[0], 0 : 0 + mat.shape[1]] += mat
    res_mat[size : size + mat.shape[0], 0 : 0 + mat.shape[1]] += mat
    res_mat[0 : 0 + mat.shape[0], size : size + mat.shape[1]] += mat
    res_mat[size : size + mat.shape[0], size : size + mat.shape[1]] += mat

    row = center_pos % size
    col = math.floor(center_pos / size)
    if row <= size / 2 and col <= size / 2:
        quad = (1, 1)
    elif row <= size / 2 and col >= size / 2:
        quad = (1, 0)
    elif row >= size / 2 and col <= size / 2:
        quad = (0, 1)
    else:
        quad = (0, 0)

    row = row + quad[0] * size
    col = col + quad[1] * size

    res_mat = res_mat[
        col - half_size : col + half_size + 1, row - half_size : row + half_size + 1
    ]

    return np.reshape(res_mat, (1, size * size))[0]


def get_next_ship_pos(ship_pos, action):
    """Returns the next position from intended action"""
    next_ship_pos = -1

    if action == 0:
        next_ship_pos = ship_pos
    elif action == 1:
        next_ship_pos = ((ship_pos) // 21) * 21 + (ship_pos + 1) % 21
    elif action == 2:
        next_ship_pos = ((ship_pos - 21) // 21) * 21 + (ship_pos) % 21
    elif action == 3:
        next_ship_pos = (((ship_pos + 21) // 21) * 21 + (ship_pos) % 21) % 441
    elif action == 4:
        next_ship_pos = ((ship_pos) // 21) * 21 + (ship_pos - 1) % 21

    return next_ship_pos
