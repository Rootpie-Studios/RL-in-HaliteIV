import math
import logging
import sys, os
import numpy as np
import scipy.optimize
from scipy.ndimage import gaussian_filter
from enum import Enum
from math import ceil
from random import random


sys.stdout = open(os.devnull, 'w')


from kaggle_environments.envs.halite.helpers import (
    Shipyard,
    Ship,
    Board,
    ShipyardAction,
    Point,
    Cell,
    ShipAction
)


logging.basicConfig(level=logging.INFO, stream=sys.stdout)

DIRECTIONS = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]
NEIGHBOURS = [Point(0, -1), Point(0, 1), Point(-1, 0), Point(1, 0)]
NEIGHBOURS2 = [
    Point(0, -1),
    Point(0, 1),
    Point(-1, 0),
    Point(1, 0),
    Point(-1, -1),
    Point(1, 1),
    Point(-1, 1),
    Point(1, -1),
]
DISTANCES = None
NAVIGATION = None
FARTHEST_DIRECTIONS_IDX = None
FARTHEST_DIRECTIONS = None
POSITIONS_IN_REACH = None
POSITIONS_IN_REACH_INDICES = None
POSITIONS_IN_TINY_RADIUS = None
POSITIONS_IN_SMALL_RADIUS = None
POSITIONS_IN_MEDIUM_RADIUS = None
SIZE = 21
TO_INDEX = {Point.from_index(index, SIZE): index for index in range(SIZE ** 2)}


def create_optimal_mining_steps_tensor(alpha, beta, gamma):
    # The optimal amount of turns spent mining on a cell based on it's distancel, the CHratio and the distance to the nearest friendly shipyard
    # Adapted from https://www.kaggle.com/solverworld/optimal-mining-with-carried-halite
    chrange = 15

    def score(n1, n2, m, H, C):
        return (
            gamma ** (n1 + m)
            * (beta * C + (1 - 0.75 ** m) * 1.02 ** (n1 + m) * H)
            / (n1 + alpha * n2 + m)
        )

    tensor = []
    for n1 in range(22):
        n_opt = []
        for n2 in range(22):
            ch_opt = []
            for ch in range(chrange):
                if ch == 0:
                    CHratio = 0
                else:
                    CHratio = math.exp((ch - 5) / 2.5)

                def h(mine):
                    return -score(n1, n2, mine, 500, CHratio * 500)

                res = scipy.optimize.minimize_scalar(
                    h, bounds=(1, 15), method="Bounded"
                )
                ch_opt.append(int(round(res.x)))
            n_opt.append(ch_opt)
        tensor.append(n_opt)
    return tensor


def compute_positions_in_reach():
    def get_in_reach(position: int):
        point = Point.from_index(position, SIZE)
        return (
            point,
            (point + NEIGHBOURS[0]) % SIZE,
            (point + NEIGHBOURS[1]) % SIZE,
            (point + NEIGHBOURS[2]) % SIZE,
            (point + NEIGHBOURS[3]) % SIZE,
        )

    def get_in_reach_indices(position: int):
        point = Point.from_index(position, SIZE)
        return np.array(
            [
                TO_INDEX[point],
                TO_INDEX[(point + NEIGHBOURS[0]) % SIZE],
                TO_INDEX[(point + NEIGHBOURS[1]) % SIZE],
                TO_INDEX[(point + NEIGHBOURS[2]) % SIZE],
                TO_INDEX[(point + NEIGHBOURS[3]) % SIZE],
            ]
        )

    global POSITIONS_IN_REACH, POSITIONS_IN_REACH_INDICES
    POSITIONS_IN_REACH = {
        Point.from_index(pos, SIZE): get_in_reach(pos) for pos in range(SIZE ** 2)
    }
    POSITIONS_IN_REACH_INDICES = np.ndarray((SIZE ** 2, 5), dtype=np.int)
    for pos in range(SIZE ** 2):  # really sad, but it's 4 am
        POSITIONS_IN_REACH_INDICES[pos] = get_in_reach_indices(pos)
    return POSITIONS_IN_REACH, POSITIONS_IN_REACH_INDICES


def get_max_distance(points):
    max_distance = 0
    for i in range(len(points)):
        pos1 = TO_INDEX[points[i]]
        for j in range(i + 1, len(points)):
            pos2 = TO_INDEX[points[j]]
            distance = get_distance(pos1, pos2)
            if distance > max_distance:
                max_distance = distance
    return max_distance


def get_blurred_halite_map(halite, sigma, multiplier=1, size=21):
    halite_map = np.array(halite).reshape((size, -1))
    blurred_halite_map = gaussian_filter(halite_map, sigma, mode="wrap")
    return multiplier * blurred_halite_map.reshape((size ** 2,))


def get_blurred_conflict_map(me, enemies, alpha, sigma, zeta, size=21):
    fight_map = np.full((size, size), fill_value=1, dtype=np.float)
    max_halite = [
        max(ship.halite for ship in player.ships) if len(player.ships) > 0 else 0
        for player in (enemies + [me])
    ]
    if len(max_halite) == 0:
        return
    max_halite = max(max_halite)
    if max_halite <= 0:
        return fight_map.reshape((size ** 2,))
    player_maps = [
        gaussian_filter(_get_player_map(player, max_halite, size), sigma, mode="wrap")
        for player in [me] + enemies
    ]
    max_value = max([np.max(player_map) for player_map in player_maps])
    for player_index, player_map in enumerate(player_maps):
        player_map = (player_map / max_value) * zeta + 1
        if player_index == 0:
            player_map *= alpha
        fight_map = np.multiply(fight_map, player_map)
    return fight_map.reshape((size ** 2,))


def _get_player_map(player, max_halite, size=21):
    player_map = np.ndarray((size ** 2,), dtype=np.float)
    for ship in player.ships:
        player_map[TO_INDEX[ship.position]] = ship.halite / max_halite
    for shipyard in player.shipyards:
        player_map[TO_INDEX[shipyard.position]] = max_halite / 2
    return player_map.reshape((size, size))


def get_cargo_map(ships, shipyards, halite_norm, size=21):
    cargo_map = np.zeros((size ** 2,), dtype=np.float)
    for ship in ships:
        cargo_map[TO_INDEX[ship.position]] += ship.halite / halite_norm
    for shipyard in shipyards:
        cargo_map[TO_INDEX[shipyard.position]] += 700 / halite_norm
    return 30 * gaussian_filter(
        cargo_map.reshape((SIZE, SIZE)), sigma=2.5, mode="wrap"
    ).reshape((-1,))


def get_hunting_matrix(ships):
    hunting_matrix = np.full(shape=(SIZE ** 2,), fill_value=99999, dtype=np.int)
    for ship in ships:
        for position in POSITIONS_IN_REACH_INDICES[TO_INDEX[ship.position]]:
            if hunting_matrix[position] > ship.halite:
                hunting_matrix[position] = ship.halite
    return hunting_matrix


def get_dominance_map(me, opponents, sigma, factor, halite_clip, size=21):
    dominance_map = np.zeros((SIZE ** 2), dtype=np.float)
    for ship in me.ships:
        dominance_map[TO_INDEX[ship.position]] += (
            clip(halite_clip - ship.halite, 0, halite_clip) / halite_clip
        )
    for shipyard in me.shipyards:
        dominance_map[TO_INDEX[shipyard.position]] += 1.5
    for player in opponents:
        for ship in player.ships:
            dominance_map[TO_INDEX[ship.position]] -= (
                clip(halite_clip - ship.halite, 0, halite_clip) / halite_clip
            )
        for shipyard in player.shipyards:
            dominance_map[TO_INDEX[shipyard.position]] -= 1.8
    blurred_dominance_map = gaussian_filter(
        dominance_map.reshape((size, size)), sigma=sigma, mode="wrap"
    )
    return factor * blurred_dominance_map.reshape((-1,))


def get_new_dominance_map(players, sigma, factor, halite_clip, size=21):
    dominance_regions = np.zeros((4, size ** 2), dtype=np.float)
    for player in players:
        player_id = player.id
        dominance_map = np.zeros((size ** 2,), dtype=np.float)
        for ship in player.ships:
            dominance_map[TO_INDEX[ship.position]] = (
                clip(halite_clip - ship.halite, 0, halite_clip) / halite_clip
            )
        for shipyard in player.shipyards:
            dominance_map[TO_INDEX[shipyard.position]] += 1.5
        dominance_regions[player_id] = factor * gaussian_filter(
            dominance_map.reshape((size, size)), sigma=sigma, mode="wrap"
        ).reshape((-1,))

    maxima = np.zeros((4, size ** 2))
    for i in range(4):
        maxima[i] = np.max(
            dominance_regions[[j for j in range(4) if j != i], :], axis=0
        )
    dominance_regions -= maxima
    return dominance_regions


def get_regions(players, sigma, halite_clip, threshold=0.1, size=21):
    dominance_map = get_new_dominance_map(players, sigma, 50, halite_clip, size)
    regions = np.full((SIZE ** 2,), fill_value=-1, dtype=np.int)
    for i in range(4):
        regions[dominance_map[i] >= threshold] = i
    return regions


def get_borders(positions):
    borders = []
    for pos in positions:
        for pos2 in get_neighbouring_positions(Point.from_index(pos, SIZE)):
            if TO_INDEX[pos2] not in positions:
                borders.append(pos)
                break
    return borders


def create_navigation_lists(size):
    """distance list taken from https://www.kaggle.com/jpmiller/fast-distance-calcs by JohnM"""
    base = np.arange(size ** 2)
    idx1 = np.repeat(base, size ** 2)
    idx2 = np.tile(base, size ** 2)

    idx_to_action = {
        0: None,
        1: ShipAction.WEST,
        2: ShipAction.EAST,
        4: ShipAction.NORTH,
        8: ShipAction.SOUTH,
    }

    idx_to_action_list = dict()
    for int_a, action_a in [(i, idx_to_action[i]) for i in range(3)]:
        for int_b, action_b in [(j, idx_to_action[j]) for j in (0, 4, 8)]:
            action_list = []
            if action_a is not None:
                action_list.append(action_a)
            if action_b is not None:
                action_list.append(action_b)
            idx_to_action_list[int_a + int_b] = action_list

    def calculate(a1, a2, smaller_val, greater_val):
        amin = np.fmin(a1, a2)
        amax = np.fmax(a1, a2)
        adiff = amax - amin
        adist = np.fmin(adiff, size - adiff)
        wrap_around = np.not_equal(adiff, adist)
        directions = np.zeros((len(a1),), dtype=np.int)
        greater = np.greater(a2, a1)
        smaller = np.greater(a1, a2)
        directions[greater != wrap_around] = greater_val
        directions[smaller != wrap_around] = smaller_val
        return adist, directions

    c1 = calculate(idx1 // size, idx2 // size, 4, 8)
    c2 = calculate(idx1 % size, idx2 % size, 1, 2)
    rowdist = c2[0]
    coldist = c1[0]
    dist_matrix = (rowdist + coldist).reshape(size ** 2, -1)

    direction_x = c2[1]
    direction_y = c1[1]
    dir_matrix = (direction_x + direction_y).reshape(size ** 2, -1)

    global DISTANCES
    DISTANCES = dist_matrix

    global NAVIGATION
    NAVIGATION = [[idx_to_action_list[a] for a in b] for b in dir_matrix]

    farthest_directions = np.zeros((size ** 4), dtype=np.int)
    farthest_directions[coldist < rowdist] += direction_x[coldist < rowdist]
    farthest_directions[coldist > rowdist] += direction_y[coldist > rowdist]
    farthest_directions[coldist == rowdist] += (
        direction_x[coldist == rowdist] + direction_y[coldist == rowdist]
    )

    global FARTHEST_DIRECTIONS_IDX
    FARTHEST_DIRECTIONS_IDX = farthest_directions.reshape((size ** 2, size ** 2))

    global FARTHEST_DIRECTIONS
    FARTHEST_DIRECTIONS = [
        [idx_to_action_list[a] for a in b] for b in FARTHEST_DIRECTIONS_IDX
    ]


def dist(a, b):
    diff = abs(a - b)
    return min(diff, SIZE - diff)


def get_axis(direction):
    if direction == ShipAction.NORTH or direction == ShipAction.SOUTH:
        return "y"
    else:
        return "x"


def get_triangles(positions, min_distance, max_distance):
    triangles = []
    if len(positions) < 3:
        return triangles
    for p1 in range(len(positions)):
        A = positions[p1]
        for p2 in range(p1 + 1, len(positions)):
            B = positions[p2]
            for p3 in range(p2 + 1, len(positions)):
                C = positions[p3]
                if is_triangle(A, B, C, min_distance, max_distance):
                    triangles.append((A, B, C))
    return triangles


def is_triangle(A, B, C, min_distance, max_distance):
    if (A.x == B.x == C.x) or (A.y == B.y == C.y):
        return False
    distances = [
        calculate_distance(A, B),
        calculate_distance(A, C),
        calculate_distance(B, C),
    ]
    if any(
        [distance < min_distance or distance > max_distance for distance in distances]
    ):
        return False
    if max(dist(A.x, B.x), dist(A.x, C.x), dist(B.x, C.x)) < 3:
        return False
    if max(dist(A.y, B.y), dist(A.y, C.y), dist(B.y, C.y)) < 3:
        return False
    return True


def create_radius_lists(tiny_radius, small_radius, medium_radius):
    global POSITIONS_IN_TINY_RADIUS, POSITIONS_IN_SMALL_RADIUS, POSITIONS_IN_MEDIUM_RADIUS
    POSITIONS_IN_TINY_RADIUS = create_radius_list(tiny_radius)
    POSITIONS_IN_SMALL_RADIUS = create_radius_list(small_radius)
    POSITIONS_IN_MEDIUM_RADIUS = create_radius_list(medium_radius)
    return (
        POSITIONS_IN_TINY_RADIUS,
        POSITIONS_IN_SMALL_RADIUS,
        POSITIONS_IN_MEDIUM_RADIUS,
    )


def create_radius_list(radius):
    radius_list = []
    for i in range(SIZE ** 2):
        radius_list.append(np.argwhere(DISTANCES[i] <= radius).reshape((-1,)).tolist())
    return radius_list


def group_ships(ships, max_group_size, max_distance):
    position_to_ship = {TO_INDEX[ship.position]: ship for ship in ships}
    groups = group_positions(
        [TO_INDEX[ship.position] for ship in ships], max_group_size, max_distance
    )
    return [[position_to_ship[position] for position in group] for group in groups]


def group_positions(positions, max_group_size, max_distance):
    groups = [[position] for position in positions]
    current_distance = 1
    while current_distance <= max_distance:
        if len(groups) <= 1:
            break
        unfinished_groups = [group for group in groups if len(group) < max_group_size]
        if len(unfinished_groups) == 0:
            break
        if min([len(group) for group in unfinished_groups]) >= math.ceil(
            max_group_size / 2
        ):
            break
        changed = True
        while changed:
            changed = False
            unfinished_groups = [
                group for group in groups if len(group) < max_group_size
            ]
            unfinished_positions = [
                position for group in unfinished_groups for position in group
            ]
            in_range = {
                position: [
                    pos2
                    for pos2 in unfinished_positions
                    if DISTANCES[position][pos2] == current_distance
                ]
                for position in unfinished_positions
            }
            position_to_group = {
                position: group_id
                for group_id, group in enumerate(groups)
                for position in group
            }
            for position, positions_in_range in in_range.items():
                if len(positions_in_range) == 0:
                    continue
                group1 = position_to_group[position]
                current_group_size = len(groups[group1])
                for pos2 in positions_in_range:
                    group2 = position_to_group[pos2]
                    if group1 == group2:
                        continue
                    if current_group_size + len(groups[group2]) <= max_group_size:
                        # merge the two groups
                        groups[group1].extend(groups[group2])
                        del groups[group2]
                        changed = True
                        break
                if changed:
                    break
        current_distance += 1
    return groups


def navigate(source: Point, target: Point, size: int):
    return NAVIGATION[TO_INDEX[source]][TO_INDEX[target]]


def nav(source: int, target: int):
    return NAVIGATION[source][target]


def get_inefficient_directions(directions):
    return [dir for dir in DIRECTIONS if dir not in directions]


def get_direction_to_neighbour(source: int, target: int) -> ShipAction:
    return NAVIGATION[source][target][0]


def calculate_distance(source: Point, target: Point):
    """
    Compute the Manhattan distance between two positions.
    :param source: The source from where to calculate
    :param target: The target to where calculate
    :return: The distance between the two positions
    """
    return DISTANCES[TO_INDEX[source]][TO_INDEX[target]]


def get_distance(source: int, target: int):
    return DISTANCES[source][target]


def get_distance_matrix():
    return DISTANCES


def get_farthest_directions_matrix():
    return FARTHEST_DIRECTIONS_IDX


def get_farthest_directions_list():
    return FARTHEST_DIRECTIONS


def get_neighbours(cell: Cell):
    return [cell.neighbor(point) for point in NEIGHBOURS]


def get_neighbouring_positions(point):
    return [(point + neighbour) % SIZE for neighbour in NEIGHBOURS]


def get_adjacent_positions(point):
    return [TO_INDEX[(point + neighbour) % SIZE] for neighbour in NEIGHBOURS2]


def get_hunting_proportion(players, halite_threshold=0):
    return [
        sum([1 for ship in player.ships if ship.halite <= halite_threshold])
        / len(player.ships)
        if len(player.ships) > 0
        else -1
        for player in players
    ]


class Vector(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        else:
            return Vector(self.x + other, self.y + other)

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        else:
            return Vector(self.x - other, self.y - other)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other)

    def __rmul__(self, other):
        return Vector(self.x * other, self.y * other)

    def __abs__(self):
        return abs(self.x) + abs(self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __mod__(self, other):
        return Vector(self.x % other, self.y % other)

    def __str__(self):
        return "(%g, %g)" % (self.x, self.y)

    def __ne__(self, other):
        return not self.__eq__(other)  # reuse __eq__


def get_excircle_midpoint(A: Point, B: Point, C: Point):
    assert A != B != C
    AB = get_vector(A, B)
    AC = get_vector(A, C)
    r = get_orthogonal_vector(AB)
    v = get_orthogonal_vector(AC)
    M1, M2 = 0.5 * Vector(AB.x, AB.y), 0.5 * Vector(AC.x, AC.y)
    a = r.y * v.x - r.x * v.y
    if a == 0:
        BC = get_vector(B, C)
        AB_abs = abs(AB)
        AC_abs = abs(AC)
        BC_abs = abs(BC)
        abs_max = max(AB_abs, AC_abs, BC_abs)
        if AB_abs == abs_max:
            return C
        elif AC_abs == abs_max:
            return B
        else:
            return A
    yololon = (M1.x * v.y - M2.x * v.y - M1.y * v.x + M2.y * v.x) / a
    Q = Vector(A.x, A.y) + M1 + yololon * r
    return Point(round(Q.x), round(Q.y)) % SIZE


def get_vector(A: Point, B: Point):
    def calculate_component(a1, a2):
        amin = min(a1, a2)
        amax = max(a1, a2)
        adiff = amax - amin
        adist = min(adiff, SIZE - adiff)
        if adiff == adist:
            return adiff if a2 == amax else -adiff
        else:
            return -adist if a2 == amax else adist

    return Vector(calculate_component(A.x, B.x), calculate_component(A.y, B.y))


def get_orthogonal_vector(v: Vector):
    return Vector(-v.y, v.x)


def clip(a, minimum, maximum):
    if a <= minimum:
        return minimum
    if a >= maximum:
        return maximum
    return a


PARAMETERS = {
    "cargo_map_halite_norm": 260,
    "cell_score_danger": 70,
    "cell_score_dominance": 0.6,
    "cell_score_enemy_halite": 0.3125712679108464,
    "cell_score_farming": -50,
    "cell_score_mine_farming": -135,
    "cell_score_neighbour_discount": 0.75,
    "cell_score_ship_halite": 0.00044866834268873795,
    "convert_when_attacked_threshold": 512,
    "disable_hunting_till": 50,
    "dominance_map_halite_clip": 110,
    "dominance_map_medium_radius": 5,
    "dominance_map_medium_sigma": 2.8668329912447335,
    "dominance_map_small_radius": 3,
    "dominance_map_small_sigma": 1.6,
    "early_second_shipyard": 30,
    "end_return_extra_moves": 3,
    "end_start": 382,
    "ending_halite_threshold": 10,
    "farming_end": 355,
    "farming_start": 30,
    "farming_start_shipyards": 2,
    "greed_min_map_diff": 14,
    "greed_stop": 30,
    "guarding_aggression_radius": 4,
    "guarding_end": 375,
    "guarding_max_distance_to_shipyard": 3,
    "guarding_max_ships_per_shipyard": 3,
    "guarding_min_distance_to_shipyard": 1,
    "guarding_norm": 1.1,
    "guarding_proportion": 0.35,
    "guarding_radius": 3,
    "guarding_radius2": 0,
    "guarding_ship_advantage_norm": 17,
    "guarding_stop": 350,
    "harvest_threshold_alpha": 0.25,
    "harvest_threshold_beta": 0.35,
    "harvest_threshold_hunting_norm": 0.65,
    "harvest_threshold_ship_advantage_norm": 15,
    "hunting_halite_threshold": 0.05,
    "hunting_max_group_distance": 5,
    "hunting_max_group_size": 1,
    "hunting_min_ships": 8,
    "hunting_proportion": 0.45,
    "hunting_proportion_after_farming": 0.4,
    "hunting_score_alpha": 0.3,
    "hunting_score_beta": 0.33026966299467286,
    "hunting_score_cargo_clip": 1.91253000892124,
    "hunting_score_delta": 0.9,
    "hunting_score_farming_position_penalty": 0.788038154434421,
    "hunting_score_gamma": 0.96,
    "hunting_score_halite_norm": 231,
    "hunting_score_hunt": 4,
    "hunting_score_intercept": 2,
    "hunting_score_iota": 0.25,
    "hunting_score_kappa": 0.15,
    "hunting_score_region": 1.3,
    "hunting_score_ship_bonus": 220,
    "hunting_score_ypsilon": 2.0795292787342774,
    "hunting_score_zeta": 0.35,
    "hunting_threshold": 11,
    "map_blur_gamma": 0.9,
    "map_blur_sigma": 0.5549610080793232,
    "map_ultra_blur": 1.75,
    "max_guarding_ships_per_target": 2,
    "max_halite_attack_shipyard": 0,
    "max_hunting_ships_per_direction": 2,
    "max_intrusion_count": 3,
    "max_ship_advantage": 13,
    "max_shipyard_distance": 8,
    "max_shipyards": 10,
    "min_enemy_shipyard_distance": 4,
    "min_mining_halite": 27,
    "min_ships": 20,
    "min_shipyard_distance": 7,
    "mining_score_alpha": 0.9981423349911503,
    "mining_score_alpha_min": 0.75,
    "mining_score_alpha_step": 0.007,
    "mining_score_beta": 0.99,
    "mining_score_beta_min": 0.93,
    "mining_score_cargo_norm": 3.5,
    "mining_score_dominance_clip": 3.3,
    "mining_score_dominance_norm": 0.5318971956494243,
    "mining_score_farming_penalty": 0.01,
    "mining_score_gamma": 0.99,
    "mining_score_juicy": 0.5,
    "mining_score_juicy_end": 0.1,
    "mining_score_minor_farming_penalty": 0.1409853963583642,
    "mining_score_start_returning": 42,
    "minor_harvest_threshold": 0.6665460554873913,
    "move_preference_base": 93,
    "move_preference_block_shipyard": -116,
    "move_preference_constructing": 138,
    "move_preference_construction_guarding": 129,
    "move_preference_guarding": 95,
    "move_preference_guarding_stay": -100,
    "move_preference_hunting": 105,
    "move_preference_longest_axis": 18,
    "move_preference_mining": 124,
    "move_preference_return": 115,
    "move_preference_stay_on_shipyard": -80,
    "return_halite": 1533,
    "second_shipyard_min_ships": 8,
    "second_shipyard_step": 18,
    "ship_spawn_threshold": 0.22612643383392095,
    "ships_shipyards_threshold": 0.17333887987761862,
    "shipyard_abandon_dominance": -20,
    "shipyard_conversion_threshold": 2.550354977134987,
    "shipyard_guarding_attack_probability": 0.5956995945455138,
    "shipyard_guarding_min_dominance": -7,
    "shipyard_min_dominance": -1,
    "shipyard_min_population": 5,
    "shipyard_min_ship_advantage": -12,
    "shipyard_start": 30,
    "shipyard_stop": 285,
    "spawn_min_dominance": -6.0,
    "spawn_till": 290,
    "third_shipyard_min_ships": 17,
    "third_shipyard_step": 58,
    "trading_start": 164,
}

EARLY_PARAMETERS = {
    "cargo_map_halite_norm": 228,
    "cell_score_danger": 70,
    "cell_score_dominance": 0.6,
    "cell_score_enemy_halite": 0.2692746373102979,
    "cell_score_farming": -50,
    "cell_score_mine_farming": -135,
    "cell_score_neighbour_discount": 0.7533825129372731,
    "cell_score_ship_halite": 0.0005,
    "convert_when_attacked_threshold": 416,
    "disable_hunting_till": 45,
    "dominance_map_halite_clip": 112,
    "dominance_map_medium_radius": 5,
    "dominance_map_medium_sigma": 2.847619968321182,
    "dominance_map_small_radius": 3,
    "dominance_map_small_sigma": 1.5287778769897649,
    "early_second_shipyard": 35,
    "end_return_extra_moves": 3,
    "end_start": 382,
    "ending_halite_threshold": 10,
    "farming_end": 355,
    "farming_start": 30,
    "farming_start_shipyards": 2,
    "greed_min_map_diff": 18,
    "greed_stop": 30,
    "guarding_aggression_radius": 4,
    "guarding_end": 375,
    "guarding_max_distance_to_shipyard": 3,
    "guarding_max_ships_per_shipyard": 3,
    "guarding_min_distance_to_shipyard": 1,
    "guarding_norm": 0.6,
    "guarding_radius": 3,
    "guarding_radius2": 0,
    "guarding_ship_advantage_norm": 17,
    "guarding_stop": 342,
    "guarding_proportion": 0.5,
    "harvest_threshold_alpha": 0.25,
    "harvest_threshold_beta": 0.35,
    "harvest_threshold_hunting_norm": 0.65,
    "harvest_threshold_ship_advantage_norm": 15,
    "hunting_halite_threshold": 0.05,
    "hunting_max_group_distance": 5,
    "hunting_max_group_size": 1,
    "hunting_min_ships": 8,
    "hunting_proportion": 0.45,
    "hunting_proportion_after_farming": 0.35,
    "hunting_score_alpha": 0.3,
    "hunting_score_beta": 0.33026966299467286,
    "hunting_score_cargo_clip": 1.6,
    "hunting_score_delta": 0.9,
    "hunting_score_farming_position_penalty": 0.8,
    "hunting_score_gamma": 0.9791318337163172,
    "hunting_score_halite_norm": 217,
    "hunting_score_hunt": 1.8482035257667,
    "hunting_score_intercept": 1.301910597972909,
    "hunting_score_iota": 0.25272451340270646,
    "hunting_score_kappa": 0.2,
    "hunting_score_region": 2.9006716643294306,
    "hunting_score_ship_bonus": 223,
    "hunting_score_ypsilon": 2.4531538731098985,
    "hunting_score_zeta": 0.35,
    "hunting_threshold": 9.886319178022347,
    "map_blur_gamma": 0.9,
    "map_blur_sigma": 0.3943145388324822,
    "map_ultra_blur": 1.75,
    "max_guarding_ships_per_target": 2,
    "max_halite_attack_shipyard": 0,
    "max_hunting_ships_per_direction": 2,
    "max_intrusion_count": 3,
    "max_ship_advantage": 15,
    "max_shipyard_distance": 8,
    "max_shipyards": 9,
    "min_enemy_shipyard_distance": 4,
    "min_mining_halite": 25,
    "min_ships": 15,
    "min_shipyard_distance": 7,
    "mining_score_alpha": 0.9597915235789294,
    "mining_score_alpha_min": 0.65,
    "mining_score_alpha_step": 0.007,
    "mining_score_beta": 0.98,
    "mining_score_beta_min": 0.93,
    "mining_score_cargo_norm": 3.5,
    "mining_score_dominance_clip": 3.477399406194472,
    "mining_score_dominance_norm": 0.2,
    "mining_score_farming_penalty": 0.01,
    "mining_score_gamma": 0.9867889914414822,
    "mining_score_juicy": 0.45,
    "mining_score_juicy_end": 0.1,
    "mining_score_minor_farming_penalty": 0.14823668299589474,
    "mining_score_start_returning": 38,
    "minor_harvest_threshold": 0.65,
    "move_preference_base": 100,
    "move_preference_block_shipyard": -116,
    "move_preference_constructing": 145,
    "move_preference_construction_guarding": 125,
    "move_preference_guarding": 102,
    "move_preference_guarding_stay": -126,
    "move_preference_hunting": 101,
    "move_preference_longest_axis": 18,
    "move_preference_mining": 121,
    "move_preference_return": 115,
    "move_preference_stay_on_shipyard": -80,
    "return_halite": 1533,
    "second_shipyard_min_ships": 8,
    "second_shipyard_step": 21,
    "ship_spawn_threshold": 0.22612643383392095,
    "ships_shipyards_threshold": 0.17333887987761862,
    "shipyard_abandon_dominance": -20,
    "shipyard_conversion_threshold": 2.550354977134987,
    "shipyard_guarding_attack_probability": 0.37557884720209034,
    "shipyard_guarding_min_dominance": -8,
    "shipyard_min_dominance": -1,
    "shipyard_min_population": 5,
    "shipyard_min_ship_advantage": -12,
    "shipyard_start": 30,
    "shipyard_stop": 285,
    "spawn_min_dominance": -6.0,
    "spawn_till": 300,
    "third_shipyard_min_ships": 16,
    "third_shipyard_step": 47,
    "trading_start": 164,
}

OPTIMAL_MINING_STEPS_TENSOR = [
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 6, 6, 5, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
        [10, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 7, 7, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
    ],
    [
        [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 6, 6, 5, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 7, 7, 6, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [10, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 7, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [12, 10, 9, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
    ],
    [
        [4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 5, 5, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 6, 6, 5, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 7, 7, 6, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [10, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 7, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [12, 10, 9, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
    ],
    [
        [4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 5, 5, 4, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 6, 6, 5, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 7, 7, 6, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [10, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 7, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [13, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
    ],
    [
        [5, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 5, 5, 4, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 6, 6, 5, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 7, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [10, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [13, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [13, 11, 10, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
    ],
    [
        [5, 4, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 5, 5, 4, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 7, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [10, 8, 8, 7, 6, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1],
        [13, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [13, 11, 10, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
    ],
    [
        [6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 5, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 6, 5, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [10, 8, 8, 7, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [13, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [13, 11, 10, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [15, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
    ],
    [
        [6, 5, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 6, 5, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 7, 6, 5, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 8, 8, 7, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [13, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [15, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [15, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
    ],
    [
        [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 6, 6, 5, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 8, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [13, 11, 10, 9, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [15, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [15, 12, 12, 11, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
    ],
    [
        [7, 6, 6, 5, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 8, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [13, 11, 10, 10, 9, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [15, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [15, 12, 12, 11, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1, 1],
    ],
    [
        [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 6, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [13, 11, 10, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [15, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 12, 12, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 14, 13, 12, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
    ],
    [
        [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [8, 7, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 6, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 7, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [13, 11, 10, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [15, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 1, 1, 1, 1, 1],
        [15, 14, 13, 12, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 15, 13, 12, 11, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
    ],
    [
        [8, 7, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1],
        [11, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [13, 11, 10, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1],
        [15, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 14, 13, 12, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 15, 13, 12, 11, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 15, 14, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
    ],
    [
        [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1],
        [11, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 9, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [13, 11, 10, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 14, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 14, 13, 12, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 15, 13, 12, 11, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 15, 14, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 15, 13, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
    ],
    [
        [9, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
        [9, 8, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [10, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 9, 9, 8, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [11, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [13, 11, 10, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 12, 11, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 14, 13, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 14, 13, 12, 11, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 15, 13, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 14, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 15, 13, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 15, 13, 12, 10, 9, 7, 5, 4, 2, 1, 1, 1, 1],
    ],
    [
        [9, 8, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [10, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [11, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 12, 11, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 14, 13, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 14, 13, 12, 11, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 15, 14, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 14, 13, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 15, 13, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 15, 13, 12, 10, 9, 7, 6, 4, 2, 1, 1, 1, 1],
        [15, 15, 15, 14, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
    ],
    [
        [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
        [10, 9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [10, 9, 9, 8, 7, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [11, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 12, 12, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 14, 13, 12, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 14, 13, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 14, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 14, 13, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 15, 13, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 15, 13, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 14, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 15, 12, 11, 9, 7, 6, 4, 3, 1, 1, 1, 1],
    ],
    [
        [10, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
        [10, 9, 9, 8, 7, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
        [11, 10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [11, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [12, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1],
        [13, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 12, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1],
        [15, 14, 13, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 14, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 14, 13, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 15, 13, 12, 10, 9, 7, 6, 4, 2, 1, 1, 1, 1],
        [15, 15, 15, 14, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 14, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 15, 12, 11, 9, 8, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 15, 13, 11, 9, 8, 6, 4, 3, 1, 1, 1, 1],
    ],
    [
        [10, 9, 9, 8, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [11, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 10, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [13, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [15, 12, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 14, 13, 12, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 14, 13, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 14, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 14, 13, 11, 10, 8, 7, 6, 4, 2, 1, 1, 1, 1],
        [15, 15, 15, 13, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 14, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 14, 12, 11, 9, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 15, 13, 11, 9, 8, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 15, 13, 11, 9, 8, 6, 5, 3, 1, 1, 1, 1],
        [15, 15, 15, 15, 13, 11, 10, 8, 6, 5, 3, 2, 1, 1, 1],
    ],
    [
        [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
        [11, 10, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [11, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 10, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [13, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [14, 12, 11, 10, 10, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 12, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1],
        [15, 14, 13, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 13, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 14, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 14, 13, 11, 10, 9, 7, 6, 4, 2, 1, 1, 1, 1],
        [15, 15, 15, 13, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 14, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 14, 12, 11, 9, 8, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 15, 13, 11, 9, 8, 6, 5, 3, 1, 1, 1, 1],
        [15, 15, 15, 15, 13, 11, 9, 8, 6, 5, 3, 1, 1, 1, 1],
        [15, 15, 15, 15, 13, 11, 10, 8, 6, 5, 3, 2, 1, 1, 1],
        [15, 15, 15, 15, 14, 12, 10, 8, 6, 5, 3, 2, 1, 1, 1],
    ],
    [
        [11, 10, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [11, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 10, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [13, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [14, 12, 11, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 12, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 14, 13, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 13, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 14, 12, 11, 10, 8, 7, 6, 4, 2, 1, 1, 1, 1],
        [15, 15, 15, 13, 11, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 13, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 14, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 14, 12, 11, 9, 8, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 15, 13, 11, 9, 8, 6, 5, 3, 1, 1, 1, 1],
        [15, 15, 15, 15, 13, 11, 10, 8, 6, 5, 3, 2, 1, 1, 1],
        [15, 15, 15, 15, 14, 11, 10, 8, 6, 5, 3, 2, 1, 1, 1],
        [15, 15, 15, 15, 14, 12, 10, 8, 7, 5, 3, 2, 1, 1, 1],
        [15, 15, 15, 15, 15, 12, 10, 8, 7, 5, 3, 2, 1, 1, 1],
    ],
    [
        [11, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
        [12, 11, 10, 9, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1],
        [12, 11, 10, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [13, 11, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
        [13, 12, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1],
        [14, 12, 11, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 12, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1],
        [15, 13, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 14, 13, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 13, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
        [15, 15, 14, 13, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 13, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 13, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 14, 12, 11, 9, 8, 6, 4, 3, 1, 1, 1, 1],
        [15, 15, 15, 15, 12, 11, 9, 8, 6, 5, 3, 1, 1, 1, 1],
        [15, 15, 15, 15, 13, 11, 9, 8, 6, 5, 3, 2, 1, 1, 1],
        [15, 15, 15, 15, 13, 11, 10, 8, 6, 5, 3, 2, 1, 1, 1],
        [15, 15, 15, 15, 14, 12, 10, 8, 7, 5, 3, 2, 1, 1, 1],
        [15, 15, 15, 15, 14, 12, 10, 8, 7, 5, 3, 2, 1, 1, 1],
        [15, 15, 15, 15, 15, 12, 10, 8, 7, 5, 4, 2, 1, 1, 1],
        [15, 15, 15, 15, 15, 12, 10, 9, 7, 5, 4, 2, 1, 1, 1],
    ],
]

SHIPS_SHIPYARDS = [0, 8, 17, 23, 28, 33, 38, 46, 49]

BOT = None


class ShipType(Enum):
    MINING = 1
    RETURNING = 2
    HUNTING = 3
    SHIPYARD_GUARDING = 4
    GUARDING = 5
    DEFENDING = 6  # guarding ships that hunt nearby enemies
    CONVERTING = 7
    CONSTRUCTING = 8  # ships on their way to a good shipyard location
    CONSTRUCTION_GUARDING = 9
    ENDING = 10


class HaliteBot(object):
    def __init__(self, parameters):
        self.late_parameters = parameters
        if EARLY_PARAMETERS is None:
            self.parameters = parameters
        else:
            self.parameters = EARLY_PARAMETERS
        self.config = None
        self.size = 21
        self.me = None
        self.player_id = 0

        self.halite = 5000
        self.ship_count = 1
        self.shipyard_count = 0
        self.shipyard_positions = []
        self.second_shipyard_ship = None
        self.first_guarding_ship = None
        self.next_shipyard_position = None
        self.blurred_halite_map = None
        self.average_halite_per_cell = 0
        self.rank = 0
        self.first_shipyard_step = 0
        self.farming_end = self.parameters["farming_end"]
        self.pseudo_shipyard = None
        self.mining_score_danger_tolerance = 10

        self.planned_moves = (
            list()
        )  # a list of positions where our ships will be in the next step
        self.planned_shipyards = list()
        self.ship_position_preferences = None
        self.ship_types = dict()
        self.mining_targets = dict()
        self.deposit_targets = dict()
        self.hunting_targets = dict()
        self.border_guards = dict()
        self.friendly_neighbour_count = dict()
        self.shipyard_guards = list()
        self.spawn_cost = 500
        self.intrusion_positions = {pos: dict() for pos in range(SIZE ** 2)}
        self.last_shipyard_count = 0

        self.enemies = list()
        self.intrusions = 0

        self.returning_ships = list()
        self.mining_ships = list()
        self.hunting_ships = list()
        self.guarding_ships = list()

        if OPTIMAL_MINING_STEPS_TENSOR is None:
            self.optimal_mining_steps = create_optimal_mining_steps_tensor(1, 1, 1)
        else:
            self.optimal_mining_steps = OPTIMAL_MINING_STEPS_TENSOR

        create_navigation_lists(self.size)
        self.distances = get_distance_matrix()
        (
            self.positions_in_reach_list,
            self.positions_in_reach_indices,
        ) = compute_positions_in_reach()
        self.farthest_directions_indices = get_farthest_directions_matrix()
        self.farthest_directions = get_farthest_directions_list()
        (
            self.tiny_radius_list,
            self.small_radius_list,
            self.medium_radius_list,
        ) = create_radius_lists(
            2,
            self.parameters["dominance_map_small_radius"],
            self.parameters["dominance_map_medium_radius"],
        )
        self.farming_radius_list = create_radius_list(
            ceil(self.parameters["max_shipyard_distance"] / 2)
        )

    def step(self, board: Board, obs):
        if self.me is None:
            self.player_id = board.current_player_id
            self.config = board.configuration
            self.size = self.config.size

        self.observation = obs
        self.me = board.current_player
        self.opponents = board.opponents
        self.ships = self.me.ships
        self.halite = self.me.halite
        self.step_count = board.step
        self.ship_count = len(self.ships)
        self.shipyard_count = len(self.me.shipyards)
        self.urgent_shipyard_guards = []
        self.shipyard_guards.clear()

        if self.step_count >= 110:
            self.parameters = self.late_parameters
        if self.step_count >= 360:
            self.mining_score_danger_tolerance = 14

        self.enemies = [
            ship
            for player in board.players.values()
            for ship in player.ships
            if player.id != self.player_id
        ]
        self.id_to_enemy = {ship.id: ship for ship in self.enemies}
        self.enemy_positions = [TO_INDEX[ship.position] for ship in self.enemies]

        self.average_halite_per_cell = (
            sum([halite for halite in self.observation["halite"]]) / self.size ** 2
        )
        self.average_halite_per_populated_cell = np.mean(
            [halite for halite in self.observation["halite"] if halite > 0]
        )
        self.average_halite_population = (
            sum([1 if halite > 0 else 0 for halite in self.observation["halite"]])
            / self.size ** 2
        )
        self.nb_cells_in_farming_radius = len(self.farming_radius_list[0])

        self.blurred_halite_map = get_blurred_halite_map(
            self.observation["halite"], self.parameters["map_blur_sigma"]
        )
        self.ultra_blurred_halite_map = get_blurred_halite_map(
            self.observation["halite"], self.parameters["map_ultra_blur"]
        )
        self.min_ultra = np.min(self.ultra_blurred_halite_map)
        self.max_ultra = np.max(self.ultra_blurred_halite_map)
        self.shipyard_positions = []
        for shipyard in self.me.shipyards:
            self.shipyard_positions.append(TO_INDEX[shipyard.position])

        players = [self.me] + self.opponents
        ranking = np.argsort(
            [self.calculate_player_score(player) for player in players]
        )[::-1]
        self.rank = int(np.where(ranking == 0)[0])
        self.player_ranking = dict()

        map_presence_ranks = np.argsort(
            [self.calculate_player_map_presence(player) for player in players]
        )[::-1]
        self.map_presence_rank = int(np.where(map_presence_ranks == 0)[0])
        self.map_presence_ranking = dict()
        map_presence = self.calculate_player_map_presence(self.me)
        self.map_presence_diff = {
            opponent.id: map_presence - self.calculate_player_map_presence(opponent)
            for opponent in self.opponents
        }

        halite_ranks = np.argsort([player.halite for player in players])[::-1]
        self.halite_ranking = dict()

        for i, player in enumerate(players):
            self.player_ranking[player.id] = int(np.where(ranking == i)[0])
            self.map_presence_ranking[player.id] = int(
                np.where(map_presence_ranks == i)[0]
            )
            self.halite_ranking[player.id] = int(np.where(halite_ranks == i)[0])

        self.ship_advantage = len(self.me.ships) - max(
            [len(player.ships) for player in self.opponents] + [0]
        )

        self.enemy_hunting_proportion = sum(
            [
                sum([1 for ship in player.ships if ship.halite <= 1])
                for player in self.opponents
                if len(player.ships) > 0
            ]
        ) / max(1, sum([len(player.ships) for player in self.opponents]))

        # shipyard connections
        self.guarded_shipyards = list()
        self.max_shipyard_connections = 0
        self.nb_connected_shipyards = 0
        for shipyard_position in self.shipyard_positions:
            connections = 0

            for shipyard2_pos in self.shipyard_positions:
                con_distance = get_distance(shipyard_position, shipyard2_pos)
                if (
                    shipyard_position != shipyard2_pos
                    and self.parameters["min_shipyard_distance"]
                    <= con_distance
                    <= self.parameters["max_shipyard_distance"]
                ):
                    connections += 1
                    if self.max_shipyard_connections < connections:
                        self.max_shipyard_connections = connections
            if connections > 0:
                self.nb_connected_shipyards += 1

        self.enemy_distances = dict()
        self.enemy_distances2 = dict()
        for pos in range(SIZE ** 2):
            min_distance = 20
            min_distance2 = 20
            for enemy_position in self.enemy_positions:
                distance = get_distance(pos, enemy_position)
                if distance < min_distance:
                    min_distance2 = min_distance
                    min_distance = distance
                elif distance < min_distance2:
                    min_distance2 = distance
            self.enemy_distances[pos] = min_distance
            self.enemy_distances2[pos] = min_distance2

        self.farming_positions = []  # Das Gelbe vom Ei
        self.minor_farming_positions = []  # Das Weiße vom Ei
        self.real_farming_points = []
        self.guarding_positions = []
        self.guarding_border = []
        if self.shipyard_count == 0:
            # There is no shipyard, but we still need to mine.
            self.shipyard_distances = [3] * self.size ** 2
        else:
            # Compute distances to the next shipyard:
            self.shipyard_distances = []
            for pos in range(0, SIZE ** 2):
                min_distance = float("inf")
                for (
                    shipyard_position
                ) in self.shipyard_positions:  # TODO: consider planned shipyards
                    distance = get_distance(pos, shipyard_position)
                    if distance < min_distance:
                        min_distance = distance
                self.shipyard_distances.append(min_distance)

        if len(self.me.ships) > 0:
            self.small_dominance_map = get_new_dominance_map(
                players,
                self.parameters["dominance_map_small_sigma"],
                22,
                self.parameters["dominance_map_halite_clip"],
            )[self.player_id]
            self.small_safety_map = get_dominance_map(
                self.me,
                self.opponents,
                self.parameters["dominance_map_small_sigma"],
                20,
                self.parameters["dominance_map_halite_clip"],
            )
            self.medium_dominance_map = get_new_dominance_map(
                players,
                self.parameters["dominance_map_medium_sigma"],
                80,
                self.parameters["dominance_map_halite_clip"],
            )[self.player_id]
            self.cargo_map = get_cargo_map(
                self.me.ships,
                self.me.shipyards,
                self.parameters["cargo_map_halite_norm"],
            )
            self.region_map = get_regions(
                players, 2.5, self.parameters["dominance_map_halite_clip"]
            )

        self.compute_regions(board)

        self.planned_moves.clear()
        self.spawn_limit_reached = self.reached_spawn_limit(board)
        self.harvest_threshold = self.calculate_harvest_threshold()
        self.positions_in_reach = []
        for ship in self.me.ships:
            self.positions_in_reach.extend(self.positions_in_reach_list[ship.position])
        self.positions_in_reach = list(set(self.positions_in_reach))
        if board.step > self.parameters["end_start"]:
            for shipyard in self.me.shipyards:
                if shipyard.position in self.positions_in_reach:
                    self.positions_in_reach.extend(
                        [shipyard.position for _ in range(3)]
                    )
        nb_positions_in_reach = len(self.positions_in_reach)
        nb_shipyard_conversions = (
            self.halite // self.config.convert_cost + 1
        )  # TODO: Ships with enough halite can also convert to a shipyard
        self.available_shipyard_conversions = (
            nb_shipyard_conversions - 1
        )  # without cargo
        self.ship_to_index = {
            ship: ship_index for ship_index, ship in enumerate(self.me.ships)
        }
        self.position_to_index = dict()
        for position_index, position in enumerate(self.positions_in_reach):
            if position in self.position_to_index.keys():
                self.position_to_index[position].append(position_index)
            else:
                self.position_to_index[position] = [position_index]

        self.danger_matrix = get_hunting_matrix(self.enemies)

        self.ship_position_preferences = np.full(
            shape=(self.ship_count, nb_positions_in_reach + nb_shipyard_conversions),
            fill_value=-999999,
        )  # positions + convert "positions"
        self.escape_matrix = np.zeros(
            (len(self.me.ships), self.size ** 2), dtype=np.float
        )
        for ship_index, ship in enumerate(self.me.ships):
            if (
                ship.halite >= self.parameters["convert_when_attacked_threshold"]
                and ship.halite + self.halite >= self.config.convert_cost
            ):
                self.ship_position_preferences[
                    ship_index, nb_positions_in_reach:
                ] = -self.parameters[
                    "convert_when_attacked_threshold"
                ]  # TODO: check whether the ship really needs to convert
            cargo = ship.halite
            danger_scores = []
            for position in self.positions_in_reach_list[ship.position]:
                pos = TO_INDEX[position]
                danger_score = 0
                for pos2 in self.small_radius_list[TO_INDEX[position]]:
                    cell_distance = get_distance(pos, pos2)
                    if cell_distance == 0:
                        discount = 1
                    elif cell_distance == 1:
                        discount = 0.75
                    elif cell_distance == 2:
                        discount = 0.45
                    else:
                        discount = 0.15
                    if self.danger_matrix[pos2] < cargo:
                        danger_score += 2 * discount
                        self.escape_matrix[
                            ship_index, self.positions_in_reach_indices[pos2]
                        ] += 1
                    elif self.danger_matrix[pos2] == cargo:
                        danger_score += discount
                        self.escape_matrix[
                            ship_index, self.positions_in_reach_indices[pos2]
                        ] += 0.5
                danger_scores.append(danger_score)
            min_danger = min(danger_scores)
            max_danger = max(danger_scores)

            danger_score = (
                self.parameters["cell_score_danger"]
                if len(
                    [
                        1
                        for danger in self.danger_matrix[
                            self.positions_in_reach_indices[TO_INDEX[ship.position]]
                        ]
                        if danger < ship.halite
                    ]
                )
                < 5
                else 3 * self.parameters["cell_score_danger"]
            )
            for pos_i, position in enumerate(
                self.positions_in_reach_list[ship.position]
            ):
                self.ship_position_preferences[
                    ship_index, self.position_to_index[position]
                ] = self.calculate_cell_score(ship, board.cells[position])
                if ship.halite > 0 and max_danger - min_danger > 1:
                    self.ship_position_preferences[
                        ship_index, self.position_to_index[position]
                    ] += int(
                        (
                            1
                            - (danger_scores[pos_i] - min_danger)
                            / (max_danger - min_danger)
                        )
                        * danger_score
                        - (danger_score // 2)
                    )
            if (
                len(
                    [
                        1
                        for pos in self.positions_in_reach_indices[
                            TO_INDEX[ship.position]
                        ]
                        if self.danger_matrix[pos] <= ship.halite
                    ]
                )
                >= 4
                and ship.halite == 0
                and board.cells[ship.position].halite > 0
                and self.step_count < 370
            ):
                self.ship_position_preferences[
                    ship_index, self.position_to_index[ship.position]
                ] -= 70

            if ship.cell.shipyard is not None:
                self.ship_position_preferences[
                    ship_index, self.position_to_index[ship.position]
                ] += (
                    self.parameters["move_preference_stay_on_shipyard"]
                    if self.step_count > (12 + self.first_shipyard_step)
                    else -200
                )  # don't block the shipyard in the early game
            # print([p for p in self.ship_position_preferences[ship_index] if p > -10000])

        self.cargo = sum([0] + [ship.halite for ship in self.me.ships])
        self.planned_shipyards.clear()
        self.border_guards.clear()
        self.ship_types.clear()
        self.mining_targets.clear()
        self.deposit_targets.clear()
        enemy_cargo = sorted([ship.halite for ship in self.enemies])
        self.hunting_halite_threshold = 0  # Always hunt with 0 halite
        self.intrusions += len(
            [
                1
                for opponent in self.opponents
                for ship in opponent.ships
                if ship.halite <= self.hunting_halite_threshold
                and TO_INDEX[ship.position] in self.farming_positions
            ]
        )
        logging.debug(
            "Total intrusions at step "
            + str(self.step_count)
            + ": "
            + str(self.intrusions)
            + " ("
            + str(round(self.intrusions / max(len(self.farming_positions), 1), 2))
            + " per position)"
        )

        for enemy in self.enemies:
            position = TO_INDEX[enemy.position]
            if position in self.farming_positions:
                if enemy.id not in self.intrusion_positions[position].keys():
                    self.intrusion_positions[position][enemy.id] = 1
                else:
                    self.intrusion_positions[position][enemy.id] += 1

        self.debug()

        self.determine_vulnerable_enemies()
        if self.step_count < self.farming_end:
            self.farming_end = self.estimate_farming_end()

        if self.handle_special_steps(board):
            return self.me.next_actions

        self.build_shipyards(board)
        self.guard_shipyards(board)

        self.spawn_cost = (
            2 * self.config.spawn_cost
            if (
                (
                    self.next_shipyard_position is not None
                    and (
                        (
                            ShipType.CONSTRUCTING in self.ship_types.values()
                            or (
                                board.cells[
                                    Point.from_index(self.next_shipyard_position, SIZE)
                                ].ship
                                is not None
                                and board.cells[
                                    Point.from_index(self.next_shipyard_position, SIZE)
                                ].ship.player_id
                                == self.player_id
                            )
                        )
                        or self.step_count < 20
                    )
                )
                and self.step_count < self.parameters["shipyard_stop"]
            )
            else self.config.spawn_cost
        )

        self.move_ships(board)
        self.spawn_ships(board)
        self.last_shipyard_count = len(self.me.shipyards)
        return self.me.next_actions

    def compute_regions(self, board: Board):
        if self.max_shipyard_connections > 0:
            triangles = get_triangles(
                [shipyard.position for shipyard in self.me.shipyards],
                self.parameters["min_shipyard_distance"],
                self.parameters["max_shipyard_distance"],
            )
            if len(triangles) > 0:
                points = triangles
            else:
                points = []
                for i in range(len(self.shipyard_positions)):
                    pos1 = self.shipyard_positions[i]
                    for j in range(i + 1, len(self.shipyard_positions)):
                        pos2 = self.shipyard_positions[j]
                        if (
                            self.parameters["min_shipyard_distance"]
                            <= get_distance(pos1, pos2)
                            <= self.parameters["max_shipyard_distance"]
                        ):
                            points.append(
                                (
                                    Point.from_index(pos1, SIZE),
                                    Point.from_index(pos2, SIZE),
                                )
                            )
                if (
                    len(points) == 1
                    and len(points[0]) == 2
                    and self.parameters["min_shipyard_distance"]
                    <= calculate_distance(points[0][0], points[0][1])
                    <= self.parameters["max_shipyard_distance"]
                ):
                    if self.pseudo_shipyard is not None:
                        points[0] = (points[0][0], points[0][1], self.pseudo_shipyard)
                    else:
                        self.pseudo_shipyard = Point.from_index(
                            self.plan_shipyard_position(True), SIZE
                        )
                        if self.pseudo_shipyard is not None:
                            points[0] = (
                                points[0][0],
                                points[0][1],
                                self.pseudo_shipyard,
                            )
        else:
            points = []

        for pos in range(SIZE ** 2):
            if (
                self.shipyard_distances[pos]
                > self.parameters["max_shipyard_distance"] + 4
            ):
                continue
            if len(points) > 0:
                for shipyard_points in points:
                    max_distance = self.parameters[
                        "max_shipyard_distance"
                    ]  # doesn't work well with the real max distance
                    guarding_radius = (
                        (max_distance + 1)
                        if self.max_shipyard_connections >= 2
                        else max_distance
                    ) + self.parameters["guarding_radius2"]
                    farming_radius = (
                        max_distance
                        if self.max_shipyard_connections >= 2
                        else max_distance - 1
                    ) - 1
                    border_radius = farming_radius + 2
                    required_in_range = min(
                        3,
                        max(
                            self.parameters["farming_start_shipyards"],
                            len(shipyard_points),
                        ),
                    )
                    if (
                        required_in_range == 2
                        and get_max_distance(shipyard_points)
                        == self.parameters["max_shipyard_distance"]
                    ):
                        guarding_radius += 1
                        farming_radius += 1
                        border_radius += 1
                    in_guarding_range = 0
                    in_farming_range = 0
                    in_minor_farming_range = 0
                    in_guarding_border = 0
                    guard = False
                    for shipyard_point in shipyard_points:
                        shipyard_pos = TO_INDEX[shipyard_point]
                        distance = get_distance(pos, shipyard_pos)
                        if (
                            distance <= self.parameters["guarding_radius"]
                            and shipyard_point != self.pseudo_shipyard
                        ):
                            guard = True
                        if distance <= border_radius:
                            in_guarding_border += 1
                        if self.parameters["farming_start"] <= self.step_count:
                            if distance <= farming_radius:
                                in_guarding_range += 1
                                in_farming_range += 1
                            elif distance <= guarding_radius:
                                in_guarding_range += 1
                            if distance <= farming_radius + 2:
                                in_minor_farming_range += 1

                    if guard or (
                        self.parameters["farming_start"] <= self.step_count
                        and in_guarding_range >= required_in_range
                    ):
                        self.guarding_positions.append(pos)
                    if (
                        pos not in self.shipyard_positions
                        and in_farming_range >= required_in_range
                    ):
                        self.farming_positions.append(pos)
                        self.guarding_border.append(pos)
                        break
                    else:
                        if (
                            pos not in self.shipyard_positions
                            and in_minor_farming_range >= required_in_range
                            and self.region_map[pos] == self.player_id
                        ):
                            self.minor_farming_positions.append(pos)
                    if in_guarding_border >= required_in_range:
                        self.guarding_border.append(pos)
            else:
                if (
                    self.shipyard_distances[pos] <= self.parameters["guarding_radius"]
                    and pos not in self.guarding_positions
                ):
                    self.guarding_positions.append(pos)
                if self.shipyard_distances[pos] == 2:
                    self.guarding_border.append(pos)

        for pos in self.farming_positions:
            point = Point.from_index(pos, SIZE)
            if board.cells[point].halite > 0:
                self.real_farming_points.append(point)
        if (
            self.step_count < self.parameters["farming_start"]
            or self.step_count > self.farming_end
        ):
            self.farming_positions = []
            self.minor_farming_positions = []
        self.guarding_positions = set(self.guarding_positions)
        self.minor_farming_positions = [
            pos
            for pos in set(self.minor_farming_positions)
            if pos not in self.farming_positions
        ]
        self.guarding_border = get_borders(set(self.guarding_border))
        self.guarding_border = [
            pos
            for pos in self.guarding_border
            if pos not in self.farming_positions and pos not in self.shipyard_positions
        ]

        changed = True
        while changed:
            changed = False
            for i in range(len(self.guarding_border)):
                position = self.guarding_border[i]
                neighbours = get_adjacent_positions(Point.from_index(position, SIZE))
                if (
                    sum(
                        [
                            1
                            for pos in neighbours
                            if pos in self.guarding_border
                            or pos in self.shipyard_positions
                        ]
                    )
                    < 2
                ):
                    self.guarding_border.remove(position)
                    changed = True
                    break

    def debug(self):
        if len(self.me.ships) > 0:
            logging.debug(
                "avg cargo at step "
                + str(self.step_count)
                + ": "
                + str(sum([ship.halite for ship in self.me.ships]) / len(self.me.ships))
            )
            if self.step_count % 25 == 0:
                map = np.zeros((SIZE ** 2,), dtype=np.int)
                for pos in range(SIZE ** 2):
                    if pos in self.guarding_positions:
                        map[pos] += 1
                    if pos in self.farming_positions:
                        map[pos] += 1
                    if pos in self.shipyard_positions:
                        map[pos] += 5
                small = self.small_dominance_map.reshape((21, 21)).round(2)
                spiegelei = np.zeros((SIZE ** 2,), dtype=np.int)
                for pos in self.farming_positions:
                    spiegelei[pos] = 2
                for pos in self.minor_farming_positions:
                    spiegelei[pos] = 1
                for pos in self.shipyard_positions:
                    spiegelei[pos] = 5
                for pos in self.guarding_positions:
                    spiegelei[pos] += 1
                for pos in self.guarding_border:
                    spiegelei[pos] = 10

    def handle_special_steps(self, board: Board) -> bool:
        step = board.step
        if step == 0:
            self.plan_shipyard_position()
        if step <= 10 and len(self.me.shipyards) == 0 and len(self.me.ships) == 1:
            ship = self.me.ships[0]
            if TO_INDEX[ship.position] != self.next_shipyard_position:
                self.ship_types[ship.id] = ShipType.CONSTRUCTING
        return False

    def plan_shipyard_position(self, preview=False):
        possible_positions = []
        if len(self.me.shipyards) == 0:
            if len(self.me.ships) == 0:
                return
            ships = self.me.ships
            ships.sort(
                key=lambda ship: self.ultra_blurred_halite_map[TO_INDEX[ship.position]],
                reverse=True,
            )
            ship_pos = TO_INDEX[ships[0].position]
            for pos in self.small_radius_list[ship_pos]:
                if self.observation["halite"][pos] <= 40:  # Don't destroy halite cells
                    possible_positions.append(
                        (
                            pos,
                            self.ultra_blurred_halite_map[pos]
                            / (1 + get_distance(ship_pos, pos)),
                        )
                    )
        elif self.max_shipyard_connections == 0:
            shipyard = self.me.shipyards[0]
            shipyard_pos = TO_INDEX[shipyard.position]
            enemy_shipyard_positions = [
                TO_INDEX[enemy_shipyard.position]
                for player in self.opponents
                for enemy_shipyard in player.shipyards
            ]
            early_second_shipyard = (
                self.step_count <= self.parameters["early_second_shipyard"]
            )
            for pos in range(SIZE ** 2):
                if (
                    self.parameters["min_shipyard_distance"]
                    <= get_distance(shipyard_pos, pos)
                    <= self.parameters["max_shipyard_distance"]
                    and min(
                        [20]
                        + [
                            get_distance(pos, enemy_pos)
                            for enemy_pos in enemy_shipyard_positions
                        ]
                    )
                    >= self.parameters["min_enemy_shipyard_distance"]
                ):
                    point = Point.from_index(pos, SIZE)
                    half = 0.5 * get_vector(shipyard.position, point)
                    half = Point(round(half.x), round(half.y))
                    midpoint = (shipyard.position + half) % SIZE
                    if early_second_shipyard and self.observation["halite"][pos] <= 50:
                        possible_positions.append(
                            (
                                pos,
                                (self.ultra_blurred_halite_map[pos] / 5)
                                + self.get_populated_cells_in_radius_count(
                                    TO_INDEX[midpoint]
                                ),
                            )
                        )
                    else:
                        possible_positions.append(
                            (
                                pos,
                                self.get_populated_cells_in_radius_count(
                                    TO_INDEX[midpoint]
                                ),
                            )
                        )
        else:
            require_dominance = self.nb_connected_shipyards > 2 and (
                self.map_presence_rank != 0
                or self.rank != 0
                or self.nb_connected_shipyards >= 4
            )
            avoid_positions = [
                TO_INDEX[enemy_shipyard.position]
                for player in self.opponents
                for enemy_shipyard in player.shipyards
                if self.map_presence_diff[player.id] < 12
            ]
            for pos in range(SIZE ** 2):
                point = Point.from_index(pos, SIZE)
                if (
                    require_dominance
                    and self.medium_dominance_map[pos]
                    < self.parameters["shipyard_min_dominance"] * 1.8
                ):
                    continue
                in_avoidance_radius = False
                if len(avoid_positions) > 0:
                    for ap in avoid_positions:
                        if (
                            get_distance(pos, ap)
                            < self.parameters["min_enemy_shipyard_distance"]
                        ):
                            in_avoidance_radius = True
                            break
                if in_avoidance_radius:
                    continue
                shipyard_distance = self.shipyard_distances[pos]
                if (
                    shipyard_distance < self.parameters["min_shipyard_distance"]
                    or self.parameters["max_shipyard_distance"] < shipyard_distance
                ):
                    continue
                point = Point.from_index(pos, SIZE)
                good_distance = [
                    shipyard.position
                    for shipyard in self.me.shipyards
                    if self.parameters["min_shipyard_distance"]
                    <= get_distance(pos, TO_INDEX[shipyard.position])
                    <= self.parameters["max_shipyard_distance"]
                ]
                if len(good_distance) >= 2:
                    for i in range(len(good_distance)):
                        pos1 = good_distance[i]
                        for j in range(i + 1, len(good_distance)):
                            pos2 = good_distance[j]
                            if is_triangle(
                                point,
                                pos1,
                                pos2,
                                self.parameters["min_shipyard_distance"],
                                self.parameters["max_shipyard_distance"],
                            ):
                                dominance = (
                                    self.medium_dominance_map[pos]
                                    if require_dominance
                                    else 0
                                )
                                midpoint = TO_INDEX[
                                    get_excircle_midpoint(pos1, pos2, point)
                                ]
                                possible_positions.append(
                                    (
                                        pos,
                                        self.get_populated_cells_in_radius_count(
                                            midpoint
                                        )
                                        + dominance,
                                    )
                                )
        if len(possible_positions) > 0:
            possible_positions.sort(key=lambda data: data[1], reverse=True)
            if not preview:
                self.next_shipyard_position = possible_positions[0][0]
                logging.info(
                    "Planning to place the next shipyard at "
                    + str(Point.from_index(self.next_shipyard_position, SIZE))
                )
            else:
                return possible_positions[0][0]

    def build_shipyards(self, board: Board):
        if len(self.me.ships) == 0:
            return
        avoid_positions = [
            TO_INDEX[enemy_shipyard.position]
            for player in self.opponents
            for enemy_shipyard in player.shipyards
            if self.map_presence_diff[player.id] < (14 if self.step_count > 130 else 10)
        ]
        in_avoidance_radius = False
        if self.next_shipyard_position is not None:
            for avoid_pos in avoid_positions:
                if (
                    get_distance(avoid_pos, self.next_shipyard_position)
                    < self.parameters["min_enemy_shipyard_distance"]
                ):
                    in_avoidance_radius = True
                    break
        if (
            self.next_shipyard_position is not None
            and not (
                self.parameters["min_shipyard_distance"]
                <= self.shipyard_distances[self.next_shipyard_position]
                <= self.parameters["max_shipyard_distance"]
                and not in_avoidance_radius
                and len(self.me.shipyards) >= self.last_shipyard_count
            )
            and len(self.me.shipyards) > 0
            and self.step_count <= self.parameters["shipyard_stop"]
        ):  # TODO: check if the position still creates a good triangle
            self.plan_shipyard_position()
        converting_disabled = (
            self.parameters["shipyard_start"] > self.step_count
            or self.step_count > self.parameters["shipyard_stop"]
        ) and (self.step_count > 10 or len(self.me.shipyards) > 0)
        if self.step_count < self.parameters["shipyard_stop"] and (
            (
                self.parameters["third_shipyard_step"] <= self.step_count < 200
                and self.max_shipyard_connections <= 1
                and self.ship_advantage > -10
                and self.ship_count >= SHIPS_SHIPYARDS[2]
            )
            or (
                self.parameters["second_shipyard_step"] <= self.step_count
                and self.max_shipyard_connections == 0
                and self.ship_advantage > -18
                and self.ship_count >= SHIPS_SHIPYARDS[1]
            )
            or (
                (
                    (
                        (max(self.nb_connected_shipyards, 1) + 1) < len(SHIPS_SHIPYARDS)
                        and len(self.me.ships)
                        >= SHIPS_SHIPYARDS[max(self.nb_connected_shipyards, 1) + 1]
                    )
                    or (
                        (max(self.nb_connected_shipyards, 1) + 1) / len(self.me.ships)
                        <= self.parameters["ships_shipyards_threshold"]
                    )
                )
                and self.ship_advantage > self.parameters["shipyard_min_ship_advantage"]
                and self.max_shipyard_connections > 1
            )
        ):
            if self.next_shipyard_position is None:
                self.plan_shipyard_position()
            elif self.small_dominance_map[self.next_shipyard_position] >= -3:
                ships = [
                    ship
                    for ship in self.me.ships
                    if (
                        ship.halite <= self.hunting_halite_threshold
                        or (
                            self.step_count <= 45
                            and self.enemy_distances[self.next_shipyard_position] >= 2
                        )
                    )
                    and ship.id not in self.ship_types.keys()
                ]
                ships.sort(
                    key=lambda ship: get_distance(
                        TO_INDEX[ship.position], self.next_shipyard_position
                    )
                )
                cell = board.cells[Point.from_index(self.next_shipyard_position, SIZE)]
                if len(ships) > 0 and (
                    cell.ship is None or cell.ship.player_id != self.player_id
                ):
                    self.ship_types[ships[0].id] = ShipType.CONSTRUCTING
                if (
                    len(ships) > 1
                    and self.halite > 250
                    and self.small_dominance_map[self.next_shipyard_position] < 3
                ):
                    self.ship_types[ships[1].id] = ShipType.CONSTRUCTION_GUARDING
            else:
                logging.debug(
                    "Dominance of "
                    + str(self.small_dominance_map[self.next_shipyard_position])
                    + " at the shipyard construction position is too low."
                )
        elif converting_disabled:
            return
        for ship in sorted(self.me.ships, key=lambda s: s.halite, reverse=True):
            if (
                self.should_convert(ship)
                and (
                    ship.id not in self.ship_types.keys()
                    or self.ship_types[ship.id] != ShipType.CONVERTING
                )
                and (
                    not converting_disabled
                    or (
                        self.next_shipyard_position is not None
                        and TO_INDEX[ship.position] == self.next_shipyard_position
                    )
                )
            ):
                self.convert_to_shipyard(ship)
                return  # only build one shipyard per step

    def spawn_ships(self, board: Board):
        # Spawn a ship if there are none left
        if (
            len(self.me.ships) == 0
            and self.halite >= self.config.spawn_cost
            and self.step_count < 390
        ):
            if len(self.me.shipyards) > 0:
                self.spawn_ship(self.me.shipyards[0])
        shipyards = self.me.shipyards
        shipyards.sort(
            key=lambda shipyard: self.calculate_spawning_score(
                TO_INDEX[shipyard.position]
            ),
            reverse=True,
        )
        for shipyard in shipyards:
            if self.halite < self.spawn_cost:  # save halite for the next shipyard
                return
            if shipyard.position in self.planned_moves:
                continue
            dominance = self.medium_dominance_map[TO_INDEX[shipyard.position]]
            if self.reached_spawn_limit(board):
                continue
            if (
                self.ship_count >= self.parameters["min_ships"]
                and self.average_halite_per_cell / self.ship_count
                < self.parameters["ship_spawn_threshold"]
            ):
                continue
            if (
                dominance < self.parameters["spawn_min_dominance"]
                and board.step > 75
                and self.shipyard_count > 1
            ):
                continue
            self.spawn_ship(shipyard)

    def reached_spawn_limit(self, board: Board):
        return board.step > self.parameters["spawn_till"] or (
            (
                self.ship_count
                >= max(
                    [
                        len(player.ships)
                        for player in board.players.values()
                        if player.id != self.player_id
                    ]
                )
                + self.parameters["max_ship_advantage"]
            )
            and self.ship_count >= self.parameters["min_ships"]
        )

    def move_ships(self, board: Board):
        if len(self.me.ships) == 0:
            return
        self.returning_ships.clear()
        self.mining_ships.clear()
        self.hunting_ships.clear()
        self.guarding_ships.clear()

        if self.shipyard_count == 0 and self.step_count > 10:
            ship = max(
                self.me.ships, key=lambda ship: ship.halite
            )  # TODO: choose the ship with the safest position
            if ship.halite + self.halite >= self.config.convert_cost:
                if (
                    self.step_count <= 385
                    or self.cargo > 800
                    or ship.halite >= self.config.convert_cost
                ):
                    self.convert_to_shipyard(ship)
                    self.ship_types[ship.id] = ShipType.CONVERTING

        ships = self.me.ships.copy()

        if (
            self.parameters["disable_hunting_till"]
            <= self.step_count
            <= (self.farming_end + 10)
        ):
            ships_for_interception = [
                ship for ship in self.me.ships if ship.id not in self.ship_types.keys()
            ]
            for target_id in self.vulnerable_ships.keys():
                safe_direction = self.vulnerable_ships[target_id]
                target = self.id_to_enemy[target_id]
                required_halite = target.halite - 1
                intercepting_ship = None
                min_halite = 9999
                min_distance = 20
                for ship in ships_for_interception:
                    if ship.id in self.ship_types.keys():
                        continue
                    if ship.halite > required_halite:
                        continue
                    if safe_direction == -2 or safe_direction == -1:
                        if self.danger_matrix[
                            TO_INDEX[ship.position]
                        ] > ship.halite and get_distance(
                            TO_INDEX[ship.position], TO_INDEX[target.position]
                        ):
                            self.ship_types[ship.id] = ShipType.HUNTING
                        continue
                    (
                        can_intercept,
                        interception_direction,
                        interception_distance,
                        interception_position,
                    ) = self.get_interception(ship, target, safe_direction)
                    if not can_intercept:
                        continue
                    if interception_distance > min_distance or (
                        interception_distance == min_distance
                        and ship.halite >= min_halite
                    ):
                        continue
                    if (
                        interception_distance > 3
                        and ship.halite > self.danger_matrix[interception_position]
                    ):
                        continue
                    intercepting_ship = ship
                    min_halite = ship.halite
                    min_distance = interception_distance
                if intercepting_ship is not None:
                    self.ship_types[intercepting_ship.id] = ShipType.HUNTING

        for ship in ships:
            ship_type = self.get_ship_type(ship, board)
            self.ship_types[ship.id] = ship_type
            if ship_type == ShipType.MINING:
                self.mining_ships.append(ship)
            elif ship_type == ShipType.RETURNING or ship_type == ShipType.ENDING:
                self.returning_ships.append(ship)
            elif ship_type == ShipType.HUNTING:
                self.hunting_ships.append(ship)

        self.assign_ship_targets(
            board
        )  # also converts some ships to hunting/returning ships

        logging.info(
            "*** Ship type breakdown for step "
            + str(self.step_count)
            + " ("
            + str(self.me.halite)
            + " halite) (avg halite: "
            + str(self.average_halite_per_populated_cell)
            + ") (ship advantage: "
            + str(self.ship_advantage)
            + ") ***"
        )
        ship_types_values = list(self.ship_types.values())
        for ship_type in set(ship_types_values):
            type_count = ship_types_values.count(ship_type)
            logging.info(
                str(ship_type).replace("ShipType.", "")
                + ": "
                + str(type_count)
                + " ("
                + str(round(type_count / len(self.me.ships) * 100, 1))
                + "%)"
            )
        while len(ships) > 0:
            ship = ships[0]
            ship_type = self.ship_types[ship.id]
            if ship_type == ShipType.MINING:
                self.handle_mining_ship(ship)
            elif ship_type == ShipType.RETURNING or ship_type == ShipType.ENDING:
                self.handle_returning_ship(ship, board)
            elif ship_type == ShipType.HUNTING or ship_type == ShipType.DEFENDING:
                self.handle_hunting_ship(ship)
            elif ship_type == ShipType.GUARDING:
                self.handle_guarding_ship(ship)
            elif ship_type == ShipType.CONSTRUCTING:
                self.handle_constructing_ship(ship)
            elif ship_type == ShipType.CONSTRUCTION_GUARDING:
                self.handle_construction_guarding_ship(ship)

            ships.remove(ship)

        row, col = scipy.optimize.linear_sum_assignment(
            self.ship_position_preferences, maximize=True
        )
        for ship_index, position_index in zip(row, col):
            ship = self.me.ships[ship_index]
            if position_index >= len(self.positions_in_reach):
                # The ship wants to convert to a shipyard
                if self.ship_position_preferences[ship_index, position_index] > 5000:
                    # immediately convert
                    ship.next_action = (
                        ShipAction.CONVERT
                    )  # self.halite has already been reduced
                else:
                    ship.next_action = ShipAction.CONVERT
                    self.halite -= self.config.convert_cost
                    self.planned_shipyards.append(ship.position)
            else:
                target = self.positions_in_reach[position_index]
                if target != ship.position:
                    ship.next_action = get_direction_to_neighbour(
                        TO_INDEX[ship.position], TO_INDEX[target]
                    )
                self.planned_moves.append(target)

    def assign_ship_targets(self, board: Board):
        # Mining assignment adapted from https://www.kaggle.com/solverworld/optimus-mine-agent
        if self.ship_count == 0:
            return

        ship_targets = {}
        mining_positions = []
        dropoff_positions = set()

        id_to_ship = {ship.id: ship for ship in self.mining_ships}

        halite_map = self.observation["halite"]
        for position, halite in enumerate(halite_map):
            if (
                halite >= self.parameters["min_mining_halite"]
                and self.small_safety_map[position] >= -2
            ):
                mining_positions.append(position)

        for shipyard in self.me.shipyards:
            shipyard_pos = TO_INDEX[shipyard.position]
            # Maybe only return to safe shipyards
            # Add each shipyard once for each distance to a ship
            for ship in self.mining_ships:
                dropoff_positions.add(
                    shipyard_pos
                    + get_distance(TO_INDEX[ship.position], shipyard_pos) * 1000
                )
        dropoff_positions = list(dropoff_positions)

        self.mining_score_beta = (
            self.parameters["mining_score_beta"]
            if self.step_count >= self.parameters["mining_score_start_returning"]
            else self.parameters["mining_score_juicy"]
        )  # Don't return too often early in the game

        if self.farming_end < self.step_count < self.parameters["end_start"]:
            self.mining_score_beta = self.parameters["mining_score_juicy_end"]

        mining_scores = np.zeros(
            (len(self.mining_ships), len(mining_positions) + len(dropoff_positions))
        )
        for ship_index, ship in enumerate(self.mining_ships):
            ship_pos = TO_INDEX[ship.position]
            general_ship_index = self.ship_to_index[ship]
            for position_index, position in enumerate(
                mining_positions + dropoff_positions
            ):
                if position >= 1000:
                    distance_to_shipyard = position // 1000
                    position = position % 1000
                    if distance_to_shipyard != get_distance(ship_pos, position):
                        mining_scores[ship_index, position_index] = -999999
                        continue

                mining_scores[ship_index, position_index] = self.calculate_mining_score(
                    general_ship_index,
                    ship_pos,
                    position,
                    halite_map[position],
                    self.blurred_halite_map[position],
                    ship.halite,
                )

        row, col = scipy.optimize.linear_sum_assignment(mining_scores, maximize=True)
        target_positions = mining_positions + dropoff_positions

        assigned_scores = [mining_scores[r][c] for r, c in zip(row, col)]
        assigned_scores.sort()
        hunting_proportion = (
            self.parameters["hunting_proportion"]
            if self.step_count < self.farming_end
            else self.parameters["hunting_proportion_after_farming"]
        )
        if len(assigned_scores) > 0:
            logging.debug(
                "assigned mining scores mean: {}".format(np.mean(assigned_scores))
            )
        hunting_enabled = board.step > self.parameters["disable_hunting_till"] and (
            self.ship_count >= self.parameters["hunting_min_ships"]
            or board.step > self.parameters["spawn_till"]
        )
        hunting_threshold = (
            max(
                np.mean(assigned_scores)
                - np.std(assigned_scores) * self.parameters["hunting_score_alpha"],
                assigned_scores[ceil(len(assigned_scores) * hunting_proportion) - 1],
            )
            if len(assigned_scores) > 0
            else -1
        )

        for r, c in zip(row, col):
            if (
                mining_scores[r][c]
                < (
                    self.parameters["hunting_threshold"]
                    if self.step_count > 80
                    else self.parameters["hunting_threshold"] - 4
                )
                or (
                    mining_scores[r][c] <= hunting_threshold
                    and self.mining_ships[r].halite <= self.hunting_halite_threshold
                )
            ) and hunting_enabled:
                continue
            if target_positions[c] >= 1000:
                ship_targets[self.mining_ships[r].id] = target_positions[c] % 1000
            else:
                ship_targets[self.mining_ships[r].id] = target_positions[c]

        # Convert indexed positions to points
        for ship_id, target_pos in ship_targets.items():
            ship = id_to_ship[ship_id]
            if target_pos in self.shipyard_positions:
                self.returning_ships.append(ship)
                self.mining_ships.remove(ship)
                self.ship_types[ship_id] = ShipType.RETURNING
                self.deposit_targets[ship_id] = Point.from_index(target_pos, self.size)
                logging.debug("Ship " + str(ship.id) + " returns.")
                continue
            self.mining_targets[ship_id] = Point.from_index(target_pos, self.size)
            logging.debug(
                "Assigning target "
                + str(Point.from_index(target_pos, self.size))
                + " to ship "
                + str(ship.id)
            )

        for ship in self.mining_ships:
            if ship.id not in self.mining_targets.keys():
                if ship.halite <= self.hunting_halite_threshold:
                    self.hunting_ships.append(ship)
                    self.ship_types[ship.id] = ShipType.HUNTING
                else:
                    self.returning_ships.append(ship)
                    self.ship_types[ship.id] = ShipType.RETURNING

        encoded_dirs = [1, 2, 4, 8]
        possible_enemy_targets = [
            (dir, ship)
            for ship in self.enemies
            for dir in encoded_dirs
            for _ in range(self.parameters["max_hunting_ships_per_direction"])
        ]
        hunting_scores = np.zeros(
            (
                len(self.hunting_ships),
                len(self.enemies)
                * 4
                * self.parameters["max_hunting_ships_per_direction"],
            )
        )
        hunting_ship_to_idx = {
            ship.id: idx for idx, ship in enumerate(self.hunting_ships)
        }
        self.interceptions = dict()
        for ship_index, ship in enumerate(self.hunting_ships):
            ship_pos = TO_INDEX[ship.position]
            for enemy_index, (direction, enemy_ship) in enumerate(
                possible_enemy_targets
            ):
                farthest_dirs = self.farthest_directions_indices[ship_pos][
                    TO_INDEX[enemy_ship.position]
                ]
                if (
                    farthest_dirs == direction
                    or (farthest_dirs - direction) in encoded_dirs
                ):
                    hunting_scores[
                        ship_index, enemy_index
                    ] = self.calculate_hunting_score(ship, enemy_ship)
                else:
                    hunting_scores[ship_index, enemy_index] = -999999

        assigned_hunting_scores = []
        row, col = scipy.optimize.linear_sum_assignment(hunting_scores, maximize=True)
        for r, c in zip(row, col):
            self.hunting_targets[self.hunting_ships[r].id] = possible_enemy_targets[c][
                1
            ]
            assigned_hunting_scores.append(hunting_scores[r, c])

        if len(self.me.shipyards) > 0:
            guarding_targets = [
                ship
                for ship in self.enemies
                if TO_INDEX[ship.position] in self.guarding_positions
            ] + [
                shipyard
                for player in self.opponents
                for shipyard in player.shipyards
                if TO_INDEX[shipyard.position] in self.guarding_positions
                and len(self.me.shipyards) < 6
            ]

            # Guarding ships
            assigned_hunting_scores.sort()
            endangered_shipyards = [
                shipyard
                for shipyard in self.me.shipyards
                if len(
                    [
                        1
                        for enemy in self.enemies
                        if get_distance(
                            TO_INDEX[enemy.position], TO_INDEX[shipyard.position]
                        )
                        <= 6
                    ]
                )
            ]
            enemies_in_guarding_zone = len(
                [
                    1
                    for e in self.enemies
                    if TO_INDEX[e.position] in self.guarding_positions
                ]
            )
            guarding_threshold_index = min(
                max(
                    min(
                        ceil(
                            (
                                0.5
                                * (
                                    1
                                    - clip(
                                        self.ship_advantage,
                                        0,
                                        self.parameters["guarding_ship_advantage_norm"],
                                    )
                                    / self.parameters["guarding_ship_advantage_norm"]
                                )
                                * (
                                    clip(
                                        self.enemy_hunting_proportion,
                                        0,
                                        self.parameters["guarding_norm"],
                                    )
                                    / self.parameters["guarding_norm"]
                                )
                                + 0.5 * self.parameters["guarding_proportion"]
                            )
                            * len(assigned_hunting_scores)
                        )
                        - 1,
                        self.parameters["guarding_max_ships_per_shipyard"]
                        * len(endangered_shipyards)
                        - 1,
                        0
                        if self.step_count >= self.parameters["guarding_end"]
                        else 500,
                    ),
                    min(len(assigned_hunting_scores) - 1, len(self.me.shipyards)),
                    len(assigned_hunting_scores)
                    - int(5 * enemies_in_guarding_zone)
                    - 1,
                )
                - len(self.guarding_ships),
                len(self.guarding_border) - 1,
            )
            if guarding_threshold_index > 0:
                guarding_threshold = assigned_hunting_scores[guarding_threshold_index]
                for r, c in zip(row, col):
                    target_pos = TO_INDEX[possible_enemy_targets[c][1].position]
                    ship_pos = TO_INDEX[self.hunting_ships[r].position]
                    if hunting_scores[r, c] < guarding_threshold:
                        if (
                            target_pos not in self.guarding_positions
                            or get_distance(ship_pos, target_pos)
                            > self.parameters["guarding_aggression_radius"]
                            or ship_pos in self.shipyard_positions
                        ):
                            self.guarding_ships.append(self.hunting_ships[r])
                        else:
                            self.ship_types[
                                self.hunting_ships[r].id
                            ] = ShipType.DEFENDING

                guarding_targets = [
                    target
                    for target in guarding_targets
                    if target not in self.hunting_targets.values()
                    or (isinstance(target, Ship) and target.halite > 0)
                ]
                unassigned_defending_ships = [
                    ship
                    for ship in self.guarding_ships
                    if ship.id not in self.shipyard_guards
                ]
                assigned_defending_ships = []
                if len(guarding_targets) > 0 and len(unassigned_defending_ships) > 0:
                    defending_targets = np.full(
                        shape=(
                            len(unassigned_defending_ships),
                            len(guarding_targets)
                            * self.parameters["max_guarding_ships_per_target"],
                        ),
                        fill_value=99999,
                        dtype=np.int,
                    )
                    for ship_index, ship in enumerate(unassigned_defending_ships):
                        ship_pos = TO_INDEX[ship.position]
                        for target_index, target in enumerate(guarding_targets):
                            distance = get_distance(ship_pos, TO_INDEX[target.position])
                            if (
                                distance
                                <= self.parameters["guarding_aggression_radius"]
                            ):
                                defending_targets[
                                    ship_index, target_index * 2 : target_index * 2 + 1
                                ] = distance
                    row, col = scipy.optimize.linear_sum_assignment(
                        defending_targets, maximize=False
                    )
                    for r, c in zip(row, col):
                        if (
                            defending_targets[r, c]
                            > self.parameters["guarding_aggression_radius"]
                        ):
                            continue
                        ship = unassigned_defending_ships[r]
                        assigned_defending_ships.append(ship.id)
                        self.ship_types[ship.id] = ShipType.DEFENDING
                        self.hunting_targets[ship.id] = guarding_targets[
                            c // self.parameters["max_guarding_ships_per_target"]
                        ]  # hunt the target (ship is still in self.guarding_ships and self.hunting_ships)

                for ship in self.guarding_ships:
                    if ship.id in assigned_defending_ships:
                        continue
                    # move to a shipyard
                    if ship in self.hunting_ships:
                        self.hunting_ships.remove(ship)
                    self.ship_types[ship.id] = ShipType.GUARDING

            self.guarding_ships = [
                ship for ship in self.guarding_ships if ship not in self.hunting_ships
            ]
            available_guarding_ships = [
                ship
                for ship in self.guarding_ships
                if ship.id not in self.shipyard_guards
            ]
            if len(available_guarding_ships) > 0:
                if len(self.guarding_border) == 0:
                    logging.error("No guarding positions")
                guarding_scores = np.zeros(
                    (len(available_guarding_ships), len(self.guarding_border))
                )
                for ship_index, ship in enumerate(available_guarding_ships):
                    ship_pos = TO_INDEX[ship.position]
                    for border_index, border_pos in enumerate(self.guarding_border):
                        guarding_scores[
                            ship_index, border_index
                        ] = self.calculate_border_score(ship_pos, border_pos)
                row, col = scipy.optimize.linear_sum_assignment(
                    guarding_scores, maximize=False
                )
                for r, c in zip(row, col):
                    self.border_guards[
                        available_guarding_ships[r].id
                    ] = self.guarding_border[c]

            for ship in [
                ship
                for ship in available_guarding_ships
                if ship.id not in self.border_guards.keys()
            ]:
                self.ship_types[ship.id] = ShipType.HUNTING
                self.hunting_ships.append(ship)

        available_hunting_ships = [
            ship
            for ship in self.hunting_ships
            if self.ship_types[ship.id] == ShipType.HUNTING
        ]
        if (
            len(available_hunting_ships) > 0
            and self.parameters["hunting_max_group_size"] > 1
        ):
            hunting_groups = group_ships(
                available_hunting_ships,
                self.parameters["hunting_max_group_size"],
                self.parameters["hunting_max_group_distance"],
            )
            hunting_group_scores = np.zeros(
                (len(hunting_groups), len(self.enemies) * 2)
            )
            step = 4 * self.parameters["max_hunting_ships_per_direction"]
            for group_idx, group in enumerate(hunting_groups):
                combined_hunting_scores = np.zeros((len(self.enemies) * 2,))
                for ship in group:
                    idx = hunting_ship_to_idx[ship.id]
                    for i in range(len(combined_hunting_scores) // 2):
                        combined_hunting_scores[i * 2 : (i + 1) * 2] += clip(
                            np.max(hunting_scores[idx, i * step : (i + 1) * step]),
                            0,
                            999999,
                        )
                combined_hunting_scores /= len(group)
                hunting_group_scores[group_idx] = combined_hunting_scores
            row, col = scipy.optimize.linear_sum_assignment(
                hunting_group_scores, maximize=True
            )
            for r, c in zip(row, col):
                for ship in hunting_groups[r]:
                    self.hunting_targets[ship.id] = possible_enemy_targets[
                        step * c // 2
                    ][1]

    def get_ship_type(self, ship: Ship, board: Board) -> ShipType:
        if ship.id in self.ship_types.keys():
            return self.ship_types[ship.id]
        ship_pos = TO_INDEX[ship.position]
        if board.step >= self.parameters["end_start"]:
            if (
                self.shipyard_distances[ship_pos]
                + board.step
                + self.parameters["end_return_extra_moves"]
                >= 398
                and ship.halite >= self.parameters["ending_halite_threshold"]
            ):
                return ShipType.ENDING
        if ship.halite >= self.parameters["return_halite"]:
            return ShipType.RETURNING
        for pos, halite in self.vulnerable_positions:
            if ship.halite <= halite and get_distance(ship_pos, pos) <= 2:
                return ShipType.HUNTING
        else:
            return ShipType.MINING

    def handle_returning_ship(self, ship: Ship, board: Board):
        if self.ship_types[ship.id] == ShipType.ENDING:
            destination = self.get_nearest_shipyard(ship.position)
            if destination is not None:
                destination = destination.position
        else:
            if ship.id in self.deposit_targets.keys():
                destination = self.deposit_targets[ship.id]
            else:
                destination = self.get_nearest_shipyard(ship.position)
                if destination is not None:
                    destination = destination.position
        if destination is None:
            if len(self.planned_shipyards) > 0:
                destination = board.cells[self.planned_shipyards[0]].position
            else:
                # TODO: mine
                logging.debug(
                    "Returning ship " + str(ship.id) + " has no shipyard to go to."
                )
                return

        ship_pos = TO_INDEX[ship.position]
        destination_pos = TO_INDEX[destination]
        if self.ship_types[ship.id] == ShipType.ENDING:
            if get_distance(ship_pos, destination_pos) == 1:
                self.change_position_score(
                    ship, destination, 9999
                )  # probably unnecessary
                logging.debug(
                    "Ending ship "
                    + str(ship.id)
                    + " returns to a shipyard at position "
                    + str(destination)
                )
                return

        self.prefer_moves(
            ship,
            navigate(ship.position, destination, self.size),
            self.farthest_directions[ship_pos][destination_pos],
            self.parameters["move_preference_return"],
            destination=destination,
        )

    def handle_mining_ship(self, ship: Ship):
        if ship.id not in self.mining_targets.keys():
            logging.error(
                "Mining ship " + str(ship.id) + " has no valid mining target."
            )
            return
        target = self.mining_targets[ship.id]
        ship_pos = TO_INDEX[ship.position]
        target_pos = TO_INDEX[target]
        reduce_farming_penalty = target_pos not in self.farming_positions
        if target != ship.position:
            self.prefer_moves(
                ship,
                nav(ship_pos, target_pos),
                self.farthest_directions[ship_pos][target_pos],
                self.parameters["move_preference_base"],
                reduce_farming_penalty=reduce_farming_penalty,
                destination=target,
            )
            if self.shipyard_distances[ship_pos] == 1:
                for neighbour in get_neighbours(ship.cell):
                    if (
                        neighbour.shipyard is not None
                        and neighbour.shipyard.player_id == self.player_id
                    ):
                        if (
                            self.step_count <= 11
                            and self.halite >= self.config.spawn_cost
                        ):
                            # We really want to get our ships out
                            self.change_position_score(
                                ship,
                                neighbour.position,
                                5 * self.parameters["move_preference_stay_on_shipyard"],
                            )
                        else:
                            self.change_position_score(
                                ship,
                                neighbour.position,
                                self.parameters["move_preference_stay_on_shipyard"],
                            )

        else:
            self.change_position_score(
                ship, target, self.parameters["move_preference_mining"]
            )
            self.prefer_moves(
                ship,
                [],
                [],
                self.parameters["move_preference_mining"],
                reduce_farming_penalty=reduce_farming_penalty,
            )

    def handle_hunting_ship(self, ship: Ship):
        ship_pos = TO_INDEX[ship.position]
        ship_position = ship.position
        penalize_farming = not (
            self.ship_types[ship.id] == ShipType.DEFENDING
            and ship.id in self.hunting_targets.keys()
            and TO_INDEX[self.hunting_targets[ship.id].position]
            in self.farming_positions
        )
        if self.step_count >= self.parameters["end_start"] and ship.halite == 0:
            enemy_shipyards = [
                shipyard
                for player in self.opponents
                for shipyard in player.shipyards
                if self.step_count + get_distance(ship_pos, TO_INDEX[shipyard.position])
                <= 398
            ]
            if len(enemy_shipyards) > 0:
                enemy_shipyards.sort(
                    key=lambda shipyard: (
                        30 - self.halite_ranking[shipyard.player_id] * 10
                        if self.halite_ranking[self.player_id] <= 1
                        else self.halite_ranking[shipyard.player_id] * 10
                    )
                    - get_distance(ship_pos, TO_INDEX[shipyard.position]),
                    reverse=True,
                )
                target = enemy_shipyards[0]
                self.prefer_moves(
                    ship,
                    navigate(ship_position, target.position, self.size),
                    self.farthest_directions[ship_pos][TO_INDEX[target.position]],
                    self.parameters["move_preference_hunting"] * 2,
                    penalize_farming,
                    destination=target.position,
                )
        if len(self.enemies) > 0:
            if ship.id in self.hunting_targets.keys():
                target = self.hunting_targets[ship.id]
            else:
                target = max(
                    self.enemies,
                    key=lambda enemy: self.calculate_hunting_score(ship, enemy),
                )
            if (
                isinstance(target, Shipyard)
                and ship.halite <= self.parameters["max_halite_attack_shipyard"]
            ) or (isinstance(target, Ship) and target.halite >= ship.halite):
                target_position = target.position
                target_pos = TO_INDEX[target_position]
                distance = get_distance(ship_pos, target_pos)
                # Don't mine
                if (
                    ship.cell.halite > 0
                    and get_distance(ship_pos, TO_INDEX[target_position]) > 1
                ):
                    self.change_position_score(
                        ship,
                        ship.position,
                        int(-0.8 * self.parameters["move_preference_hunting"]),
                    )
                if (ship.id + target.id) in self.interceptions.keys():
                    self.prefer_moves(
                        ship,
                        self.interceptions[ship.id + target.id],
                        [],
                        self.parameters["move_preference_hunting"],
                        penalize_farming,
                        destination=target.position,
                    )
                else:
                    if distance > 2 or isinstance(target, Shipyard):
                        self.prefer_moves(
                            ship,
                            navigate(ship_position, target_position, self.size),
                            self.farthest_directions[ship_pos][
                                TO_INDEX[target_position]
                            ],
                            self.parameters["move_preference_hunting"],
                            penalize_farming,
                            destination=target.position,
                        )
                    else:
                        possible_target_positions = self.positions_in_reach_list[
                            target_position
                        ]
                        possible_moves = []
                        for position in self.positions_in_reach_list[ship.position]:
                            if position not in possible_target_positions:
                                if position != ship.position or self.observation[
                                    "halite"
                                ][TO_INDEX[position]] >= 4 * (
                                    target.halite - ship.halite
                                ):
                                    self.change_position_score(
                                        ship,
                                        position,
                                        int(
                                            -self.parameters["move_preference_hunting"]
                                            // 1.2
                                        ),
                                    )
                                else:
                                    self.change_position_score(
                                        ship,
                                        position,
                                        int(
                                            -self.parameters["move_preference_hunting"]
                                            // 2
                                        ),
                                    )
                            else:
                                possible_moves.append(position)
                        for pos_idx in range(len(possible_moves)):
                            if (
                                target.id + str(TO_INDEX[possible_moves[pos_idx]] + 1)
                            ) not in self.escape_count.keys():
                                self.escape_count[
                                    target.id
                                    + str(TO_INDEX[possible_moves[pos_idx]] + 1)
                                ] = 3
                                logging.critical(
                                    "Escape position "
                                    + str(possible_moves[pos_idx])
                                    + " for ship "
                                    + str(ship.id)
                                    + " not in escape keys: "
                                    + str(
                                        target.id
                                        + str(TO_INDEX[possible_moves[pos_idx]] + 1)
                                    )
                                )
                        move_ranks = dict()
                        for i, count in enumerate(
                            sorted(
                                set(
                                    [
                                        self.escape_count[
                                            target.id + str(TO_INDEX[pos] + 1)
                                        ]
                                        for pos in possible_moves
                                    ]
                                ),
                                reverse=True,
                            )
                        ):
                            move_ranks[count] = i
                        for pos in possible_moves:
                            if pos != ship.position or self.observation["halite"][
                                TO_INDEX[pos]
                            ] < 4 * (target.halite - ship.halite):
                                self.change_position_score(
                                    ship,
                                    pos,
                                    int(
                                        self.parameters["move_preference_hunting"]
                                        / (
                                            move_ranks[
                                                self.escape_count[
                                                    target.id + str(TO_INDEX[pos] + 1)
                                                ]
                                            ]
                                            + 1
                                        )
                                    ),
                                )
                            elif (
                                target.id in self.vulnerable_ships.keys()
                                and self.vulnerable_ships[target.id] == -2
                            ):
                                self.change_position_score(
                                    ship,
                                    pos,
                                    self.parameters["move_preference_hunting"],
                                )
                            else:
                                self.change_position_score(
                                    ship,
                                    pos,
                                    int(-self.parameters["move_preference_hunting"]),
                                )

    def handle_guarding_ship(self, ship: Ship):
        if ship.id not in self.border_guards.keys():
            logging.error(
                "Guarding ship " + str(ship.id) + " has no shipyard to guard."
            )
            return
        ship_pos = TO_INDEX[ship.position]
        target_position = self.border_guards[ship.id]
        if ship.id in self.shipyard_guards:
            if ship_pos != target_position:
                move_preference = (
                    self.parameters["move_preference_guarding"] * 2
                    if get_distance(ship_pos, target_position) > 3
                    or ship.id in self.urgent_shipyard_guards
                    else self.parameters["move_preference_guarding"]
                )
                self.prefer_moves(
                    ship,
                    nav(ship_pos, target_position),
                    self.farthest_directions[ship_pos][target_position],
                    move_preference,
                    False,
                    destination=Point.from_index(target_position, SIZE),
                )
            else:
                self.change_position_score(
                    ship,
                    ship.position,
                    self.parameters["move_preference_guarding"]
                    - self.parameters["move_preference_stay_on_shipyard"],
                )
        else:
            # stay near the shipyard
            if ship.cell.halite > 0:
                self.change_position_score(
                    ship,
                    ship.position,
                    self.parameters["move_preference_guarding_stay"],
                )
            self.prefer_moves(
                ship,
                nav(ship_pos, target_position),
                self.farthest_directions[ship_pos][target_position],
                self.parameters["move_preference_guarding"],
                False,
                destination=Point.from_index(target_position, SIZE),
            )

    def handle_constructing_ship(self, ship: Ship):
        if self.next_shipyard_position is None:
            if len(self.planned_shipyards) == 0:
                logging.error(
                    "Constructing ship " + str(ship.id) + " has no construction target."
                )
                return
            else:
                shipyard_point = self.planned_shipyards[0]
        else:
            shipyard_point = Point.from_index(self.next_shipyard_position, SIZE)
        logging.debug(
            "Constructing ship "
            + str(ship.id)
            + " at position "
            + str(ship.position)
            + " is on the way to "
            + str(shipyard_point)
            + "."
        )
        self.prefer_moves(
            ship,
            navigate(ship.position, shipyard_point, self.size),
            self.farthest_directions[TO_INDEX[ship.position]][
                self.next_shipyard_position
            ],
            self.parameters["move_preference_constructing"],
            destination=shipyard_point,
        )

    def handle_construction_guarding_ship(self, ship: Ship):
        if self.next_shipyard_position is None:
            if len(self.planned_shipyards) == 0:
                logging.error(
                    "Construction guarding ship  "
                    + str(ship.id)
                    + " has no construction target."
                )
                return
            else:
                shipyard_point = self.planned_shipyards[0]
        else:
            shipyard_point = Point.from_index(self.next_shipyard_position, SIZE)
        self.prefer_moves(
            ship,
            navigate(ship.position, shipyard_point, self.size),
            self.farthest_directions[TO_INDEX[ship.position]][TO_INDEX[shipyard_point]],
            self.parameters["move_preference_construction_guarding"],
            destination=shipyard_point,
        )

    def guard_shipyards(self, board: Board):
        shipyard_guards = []
        if len(self.planned_shipyards) > 0:
            for shipyard_point in self.planned_shipyards:
                self.guard_position(TO_INDEX[shipyard_point], shipyard_guards)
        for shipyard in self.me.shipyards:
            if shipyard.position in self.planned_moves:
                continue
            shipyard_position = TO_INDEX[shipyard.position]
            dominance = self.medium_dominance_map[shipyard_position]

            min_distance = self.guard_position(shipyard_position, shipyard_guards)
            enemies = set(
                filter(
                    lambda cell: cell.ship is not None
                    and cell.ship.player_id != self.player_id,
                    get_neighbours(shipyard.cell),
                )
            )
            max_halite = (
                min([cell.ship.halite for cell in enemies]) if len(enemies) > 0 else 500
            )

            if len(enemies) > 0 and min_distance != 1:
                # TODO: maybe don't move on the shipyard if the dominance score is too low
                if shipyard.cell.ship is not None:
                    self.ship_types[shipyard.cell.ship.id] = ShipType.SHIPYARD_GUARDING
                    if (
                        self.halite < self.spawn_cost
                        or (
                            self.step_count > self.parameters["spawn_till"]
                            and (self.shipyard_count > 1 or self.step_count > 385)
                        )
                        or dominance
                        < self.parameters["shipyard_guarding_min_dominance"]
                        or random()
                        > self.parameters["shipyard_guarding_attack_probability"]
                        or self.step_count >= self.parameters["guarding_stop"]
                    ):
                        if dominance > self.parameters["shipyard_abandon_dominance"]:
                            self.change_position_score(
                                shipyard.cell.ship, shipyard.cell.position, 10000
                            )
                            logging.debug(
                                "Ship "
                                + str(shipyard.cell.ship.id)
                                + " stays at position "
                                + str(shipyard.position)
                                + " to guard a shipyard."
                            )
                    else:
                        self.spawn_ship(shipyard)
                        for enemy in enemies:
                            logging.debug("Attacking a ship near our shipyard")
                            self.change_position_score(
                                shipyard.cell.ship, enemy.position, 500
                            )  # equalize to crash into the ship even if that means we also lose our ship
                            self.attack_position(
                                enemy.position
                            )  # Maybe also do this if we don't spawn a ship, but can move one to the shipyard
                else:
                    potential_guards = [
                        neighbour.ship
                        for neighbour in get_neighbours(shipyard.cell)
                        if neighbour.ship is not None
                        and neighbour.ship.player_id == self.player_id
                        and neighbour.ship.halite <= max_halite
                    ]
                    if len(potential_guards) > 0 and (
                        self.reached_spawn_limit(board) or self.halite < self.spawn_cost
                    ):
                        guard = sorted(potential_guards, key=lambda ship: ship.halite)[
                            0
                        ]
                        self.change_position_score(guard, shipyard.position, 8000)
                        self.ship_types[guard.id] = ShipType.SHIPYARD_GUARDING
                        logging.debug(
                            "Ship "
                            + str(guard.id)
                            + " moves to position "
                            + str(shipyard.position)
                            + " to protect a shipyard."
                        )
                    elif (
                        self.halite > self.spawn_cost
                        and (
                            dominance
                            >= self.parameters["shipyard_guarding_min_dominance"]
                            or board.step <= 25
                            or self.shipyard_count == 1
                        )
                        and (
                            self.step_count < self.parameters["guarding_stop"]
                            or (
                                self.shipyard_count == 1
                                and self.step_count < self.parameters["end_start"]
                            )
                        )
                    ):
                        logging.debug(
                            "Shipyard "
                            + str(shipyard.id)
                            + " spawns a ship to defend the position."
                        )
                        self.spawn_ship(shipyard)
                    else:
                        logging.info(
                            "Shipyard " + str(shipyard.id) + " cannot be protected."
                        )

        self.shipyard_guards = list(set(self.shipyard_guards))
        self.guarding_ships = list(set(self.guarding_ships))

    def guard_position(self, shipyard_position, shipyard_guards):
        enemy_distance = self.enemy_distances[shipyard_position]
        enemy_distance2 = self.enemy_distances2[shipyard_position]
        dominance = self.medium_dominance_map[shipyard_position]
        min_distance = 20
        min_distance2 = 20
        guard = None
        guard2 = None
        for ship in [
            ship
            for ship in self.me.ships
            if (
                ship.id not in self.ship_types.keys()
                or self.ship_types[ship.id]
                not in [ShipType.CONVERTING, ShipType.CONSTRUCTING]
            )
            and ship.id not in shipyard_guards
        ]:
            distance = get_distance(shipyard_position, TO_INDEX[ship.position])
            if distance < min_distance and (
                ship.halite <= self.hunting_halite_threshold
                or distance < enemy_distance
            ):
                min_distance2 = min_distance
                min_distance = distance
                guard2 = guard
                guard = ship
            elif distance < min_distance2 and (
                ship.halite <= self.hunting_halite_threshold
                or distance < enemy_distance2
            ):
                guard2 = ship
                min_distance2 = distance

        if guard is not None:
            shipyard_guards.append(guard.id)  # don't append guard2
        if dominance < self.parameters["shipyard_abandon_dominance"]:
            logging.debug(
                "Abandoning shipyard at position "
                + str(Point.from_index(shipyard_position, SIZE))
            )
        elif enemy_distance - 1 <= min_distance and guard is not None:
            self.shipyard_guards.append(guard.id)
            self.guarding_ships.append(guard)
            self.border_guards[guard.id] = shipyard_position
            self.ship_types[guard.id] = ShipType.GUARDING
            if enemy_distance <= min_distance:
                self.urgent_shipyard_guards.append(guard.id)
        elif enemy_distance - 2 <= min_distance and guard is not None:
            if guard.cell.halite > 0:
                self.change_position_score(guard, guard.position, -500)  # don't mine

        if dominance < self.parameters["shipyard_abandon_dominance"]:
            logging.debug(
                "Abandoning shipyard at position "
                + str(Point.from_index(shipyard_position, SIZE))
            )
        elif enemy_distance2 - 1 <= min_distance2 and guard2 is not None:
            self.shipyard_guards.append(guard2.id)
            self.guarding_ships.append(guard2)
            self.border_guards[guard2.id] = shipyard_position
            self.ship_types[guard2.id] = ShipType.GUARDING
            if enemy_distance2 <= min_distance2:
                self.urgent_shipyard_guards.append(guard2.id)
        elif enemy_distance2 - 2 <= min_distance2 and guard2 is not None:
            if guard2.cell.halite > 0:
                self.change_position_score(guard2, guard2.position, -500)  # don't mine
        return min_distance

    def determine_vulnerable_enemies(self):
        hunting_matrix = get_hunting_matrix(self.me.ships)
        self.vulnerable_ships = dict()
        self.escape_count = dict()
        self.vulnerable_positions = []
        for ship in self.enemies:
            ship_pos = TO_INDEX[ship.position]
            escape_positions = []
            for pos in self.positions_in_reach_indices[ship_pos]:
                escape_possibilities = []
                for pos2 in self.positions_in_reach_indices[pos]:
                    if pos2 != pos and hunting_matrix[pos2] >= ship.halite:
                        escape_possibilities.append(pos2)
                escape_possibilities = len(escape_possibilities)
                if hunting_matrix[pos] >= ship.halite and escape_possibilities > 1:
                    escape_positions.append(pos)
                self.escape_count[ship.id + str(pos + 1)] = escape_possibilities
            if len(escape_positions) == 0:
                self.vulnerable_ships[ship.id] = -2
                self.vulnerable_positions.append((TO_INDEX[ship.position], ship.halite))
            elif len(escape_positions) == 1:
                if escape_positions[0] == ship_pos:
                    self.vulnerable_ships[ship.id] = -1  # stay still
                else:
                    self.vulnerable_ships[ship.id] = get_direction_to_neighbour(
                        ship_pos, escape_positions[0]
                    )
                self.vulnerable_positions.append((TO_INDEX[ship.position], ship.halite))
        logging.info("Number of vulnerable ships: " + str(len(self.vulnerable_ships)))

    def should_convert(self, ship: Ship):
        if self.halite + ship.halite < self.config.convert_cost:
            return False
        ship_pos = TO_INDEX[ship.position]
        min_distance = 20
        for enemy_pos in self.enemy_positions:
            if get_distance(ship_pos, enemy_pos) < min_distance:
                min_distance = get_distance(ship_pos, enemy_pos)
        guards = [
            1
            for guard in self.me.ships
            if 0 < get_distance(TO_INDEX[guard.position], ship_pos) < min_distance
            or (
                get_distance(TO_INDEX[guard.position], ship_pos) == 1
                and guard.id in self.ship_types.keys()
                and self.ship_types[guard.id] == ShipType.CONSTRUCTION_GUARDING
            )
        ]
        if len(guards) == 0 and self.ship_count > 5:
            return False
        if (
            ship_pos == self.next_shipyard_position
            and self.step_count <= self.parameters["shipyard_stop"]
        ):
            return True
        if self.shipyard_count == 0 and self.step_count <= 10:
            return False
        if self.shipyard_count == 0 and (
            self.step_count <= self.parameters["end_start"]
            or ship.halite >= self.config.convert_cost
            or self.cargo >= 1200
        ):
            return True  # TODO: choose best ship
        if self.nb_connected_shipyards >= self.parameters["max_shipyards"]:
            return False
        if (
            self.average_halite_per_cell / self.shipyard_count
            < self.parameters["shipyard_conversion_threshold"]
            or (max(self.nb_connected_shipyards, 1) + 1) / self.ship_count
            >= self.parameters["ships_shipyards_threshold"]
        ):
            return False
        if (
            self.medium_dominance_map[ship_pos]
            < self.parameters["shipyard_min_dominance"]
        ):
            return False
        return self.creates_good_triangle(ship.position)

    def creates_good_triangle(self, point):
        ship_pos = TO_INDEX[point]
        distance_to_nearest_shipyard = self.shipyard_distances[ship_pos]
        if (
            self.parameters["min_shipyard_distance"]
            <= distance_to_nearest_shipyard
            <= self.parameters["max_shipyard_distance"]
        ):
            good_distance = []
            for shipyard_position in self.shipyard_positions:
                if (
                    self.parameters["min_shipyard_distance"]
                    <= get_distance(ship_pos, shipyard_position)
                    <= self.parameters["max_shipyard_distance"]
                ):
                    good_distance.append(Point.from_index(shipyard_position, SIZE))
            if len(good_distance) == 0:
                return False
            midpoints = []
            if self.max_shipyard_connections == 0:
                half = 0.5 * get_vector(point, good_distance[0])
                half = Point(round(half.x), round(half.y))
                midpoints.append((point + half) % SIZE)
            else:
                for i in range(len(good_distance)):
                    for j in range(i + 1, len(good_distance)):
                        pos1, pos2 = good_distance[i], good_distance[j]
                        if (pos1.x == pos2.x == point.x) or (
                            pos1.y == pos2.y == point.y
                        ):  # rays don't intersect
                            midpoints.append(point)
                        else:
                            midpoints.append(get_excircle_midpoint(pos1, pos2, point))
            threshold = (
                self.parameters["shipyard_min_population"]
                * self.average_halite_population
                * self.nb_cells_in_farming_radius
            )
            if any(
                [
                    self.get_populated_cells_in_radius_count(TO_INDEX[midpoint])
                    >= threshold
                    for midpoint in set(midpoints)
                ]
            ):
                return True
        return False

    def get_populated_cells_in_radius_count(self, position):
        return sum(
            [
                1 if self.observation["halite"][cell] > 0 else 0
                for cell in self.farming_radius_list[position]
            ]
        )

    def calculate_mining_score(
        self,
        ship_index,
        ship_position: int,
        cell_position: int,
        halite,
        blurred_halite,
        ship_halite,
        debug=False,
    ) -> float:
        distance_from_ship = get_distance(ship_position, cell_position)
        distance_from_shipyard = self.shipyard_distances[cell_position]
        # mining_score_alpha = clip(self.parameters['mining_score_alpha'] * ship_halite / (self.parameters['mining_score_cargo_norm'] * self.average_halite_per_populated_cell), self.parameters['mining_score_alpha_min'] + self.parameters['mining_score_alpha_step'] * self.step_count, self.parameters['mining_score_alpha'])
        mining_score_alpha = (
            self.parameters["mining_score_alpha"]
            if self.step_count > 20 + distance_from_shipyard
            else 0.5
        )
        mining_score_beta = self.mining_score_beta

        position_quality = self.ultra_blurred_halite_map[ship_position] / self.max_ultra
        if mining_score_alpha == self.parameters["mining_score_alpha"]:
            mining_score_alpha *= self.parameters["mining_score_alpha_min"] + (
                1 - self.parameters["mining_score_alpha_min"]
            ) * (1 - position_quality)
        if mining_score_beta == self.parameters["mining_score_beta"]:
            mining_score_beta *= self.parameters["mining_score_beta_min"] + (
                1 - self.parameters["mining_score_beta_min"]
            ) * (1 - position_quality)

        halite_val = (
            1 - self.parameters["map_blur_gamma"] ** distance_from_ship
        ) * blurred_halite + self.parameters[
            "map_blur_gamma"
        ] ** distance_from_ship * halite
        if cell_position in self.enemy_positions and distance_from_ship > 1:
            halite_val *= 0.75 ** (distance_from_ship - 1)
        else:
            halite_val = min(1.02 ** distance_from_ship * halite_val, 500)
        farming_activated = (
            self.parameters["farming_start"]
            <= (self.step_count + distance_from_ship)
            < self.farming_end
        )
        if distance_from_shipyard > 20:
            # There is no shipyard.
            distance_from_shipyard = 20
        if ship_halite == 0:
            ch = 0
        elif halite_val == 0:
            ch = 14
        else:
            ch = clip(
                int(math.log(mining_score_beta * ship_halite / halite_val) * 2.5 + 5.5),
                0,
                14,
            )
        if distance_from_shipyard == 0:
            mining_steps = 0
            if distance_from_ship == 0:
                return 0  # We are on the shipyard
        elif (
            cell_position in self.farming_positions
            and halite >= self.harvest_threshold
            and farming_activated
        ):  # halite not halite_val because we cannot be sure the cell halite regenerates
            mining_steps = ceil(math.log(self.harvest_threshold / halite_val, 0.75))
        else:
            mining_steps = self.optimal_mining_steps[max(distance_from_ship, 0)][
                max(int(round(mining_score_alpha * distance_from_shipyard)), 0)
            ][ch]
        if self.step_count >= self.parameters["end_start"]:
            ending_steps = (
                self.step_count
                + distance_from_ship
                + mining_steps
                + distance_from_shipyard
                + self.parameters["end_return_extra_moves"] // 2
                - 398
            )
            if ending_steps > 0:
                mining_steps = max(mining_steps - ending_steps, 0)
        safety = (
            self.parameters["mining_score_dominance_norm"]
            * clip(
                self.small_safety_map[cell_position]
                + self.parameters["mining_score_dominance_clip"],
                0,
                1.5 * self.parameters["mining_score_dominance_clip"],
            )
            / (1.5 * self.parameters["mining_score_dominance_clip"])
        )
        if self.step_count < self.parameters["mining_score_start_returning"]:
            safety /= 1.5
            safety += self.parameters["mining_score_dominance_norm"] / 3
        safety += 1 - self.parameters["mining_score_dominance_norm"] / 2
        if debug:
            print(mining_score_alpha)
            print(mining_score_beta)
        score = (
            self.parameters["mining_score_gamma"] ** (distance_from_ship + mining_steps)
            * (
                mining_score_beta * ship_halite
                + (1 - 0.75 ** mining_steps) * halite_val
            )
            * safety
            / max(
                distance_from_ship
                + mining_steps
                + mining_score_alpha * distance_from_shipyard,
                1,
            )
        )
        if distance_from_shipyard == 0 and self.step_count <= 11:
            score *= 0.1  # We don't want to block the shipyard.
        if farming_activated:
            if (
                halite < self.harvest_threshold
                and cell_position in self.farming_positions
            ):  # halite not halite_val because we cannot be sure the cell halite regenerates
                score *= self.parameters["mining_score_farming_penalty"]
            elif (
                halite
                < self.parameters["minor_harvest_threshold"] * self.harvest_threshold
                and cell_position in self.minor_farming_positions
            ):
                score *= self.parameters["mining_score_minor_farming_penalty"]
        if (
            self.step_count <= 8 + self.first_shipyard_step
            and distance_from_shipyard <= 3
        ):
            score *= 0.05
        escape_score = (
            1
            - clip(
                self.escape_matrix[ship_index, cell_position]
                - self.mining_score_danger_tolerance,
                0,
                22,
            )
            / 22
        )
        score *= escape_score
        if (
            escape_score > 0.85
            and min(self.danger_matrix[self.tiny_radius_list[cell_position]])
            <= ship_halite
            and self.step_count > self.parameters["greed_stop"]
        ):
            score *= clip(
                min(self.danger_matrix[self.tiny_radius_list[cell_position]])
                / (ship_halite + 10),
                0.25,
                0.75,
            )
        return score

    def calculate_hunting_score(self, ship: Ship, enemy: Ship) -> float:
        d_halite = enemy.halite - ship.halite
        ship_pos = TO_INDEX[ship.position]
        enemy_pos = TO_INDEX[enemy.position]
        distance = get_distance(ship_pos, enemy_pos)
        if d_halite < 0:
            halite_score = -9999
        elif d_halite == 0:
            halite_score = (
                0.25
                * self.parameters["hunting_score_ship_bonus"]
                * (1 - self.step_count / 398)
                / self.parameters["hunting_score_halite_norm"]
            )
        else:
            ship_bonus = self.parameters["hunting_score_ship_bonus"] * (
                1 - self.step_count / 398
            )
            halite_score = (ship_bonus + d_halite) / self.parameters[
                "hunting_score_halite_norm"
            ]
        dominance_influence = (
            self.parameters["hunting_score_delta"]
            + self.parameters["hunting_score_beta"]
            * clip(self.medium_dominance_map[enemy_pos] + 10, 0, 20)
            / 20
        )
        player_score = 1 + self.parameters["hunting_score_kappa"] * (
            3 - self.player_ranking[ship.player_id]
            if self.rank <= 1
            else self.player_ranking[ship.player_id]
        )
        score = (
            self.parameters["hunting_score_gamma"] ** distance
            * halite_score
            * (
                self.parameters["hunting_score_region"]
                if enemy_pos in self.guarding_positions
                else 1
            )
            * dominance_influence
            * player_score
            * (
                1
                + (
                    self.parameters["hunting_score_iota"]
                    * clip(self.ultra_blurred_halite_map[enemy_pos], 0, 200)
                    / 200
                )
            )
            * (
                1
                + (
                    self.parameters["hunting_score_zeta"]
                    * clip(
                        self.cargo_map[enemy_pos],
                        0,
                        self.parameters["hunting_score_cargo_clip"],
                    )
                    / self.parameters["hunting_score_cargo_clip"]
                )
            )
        )
        if (
            self.next_shipyard_position is not None
            and get_distance(enemy_pos, self.next_shipyard_position) <= 3
        ):
            # Clear space for a shipyard
            score *= self.parameters["hunting_score_ypsilon"]

        if enemy.id in self.vulnerable_ships.keys():
            safe_direction = self.vulnerable_ships[enemy.id]
            if distance <= 2:
                score *= self.parameters["hunting_score_hunt"]
            elif (
                safe_direction != -1 and safe_direction != -2
            ):  # The ship can only stay at it's current position or it has no safe position.
                can_intercept, target_dir, _, _ = self.get_interception(
                    ship, enemy, safe_direction
                )
                if can_intercept:
                    score *= self.parameters["hunting_score_intercept"]
                    self.interceptions[ship.id + enemy.id] = target_dir
            elif distance <= 4:
                score *= self.parameters["hunting_score_intercept"] / 2
        if (
            len(self.real_farming_points) > 0
            and enemy_pos not in self.farming_positions
        ):
            farming_positions_in_the_way = min(
                [
                    self.get_farming_positions_count_in_between(
                        ship.position, enemy.position, dir
                    )
                    for dir in nav(ship_pos, enemy_pos)
                ]
            )
            score *= (
                self.parameters["hunting_score_farming_position_penalty"]
                ** farming_positions_in_the_way
            )
        return score

    def get_interception(self, ship, enemy, safe_direction):
        ship_pos = TO_INDEX[ship.position]
        enemy_pos = TO_INDEX[enemy.position]
        if safe_direction == ShipAction.WEST or safe_direction == ShipAction.EAST:
            # chase along the x-axis
            interception_pos = TO_INDEX[Point(ship.position.x, enemy.position.y)]
        else:
            # chase along the y-axis
            interception_pos = TO_INDEX[Point(enemy.position.x, ship.position.y)]
        target_dir = nav(enemy_pos, interception_pos)
        interception_distance = get_distance(ship_pos, interception_pos)
        interception_direction = nav(ship_pos, interception_pos)
        if len(interception_direction) == 0 and get_distance(ship_pos, enemy_pos) > 1:
            interception_direction = nav(ship_pos, enemy_pos)
        if (len(target_dir) > 0 and target_dir[0] == safe_direction) and get_distance(
            enemy_pos, interception_pos
        ) >= interception_distance:
            # We can intercept the target
            return True, interception_direction, interception_distance, interception_pos
        else:
            return False, None, 99, None

    def calculate_cell_score(self, ship: Ship, cell: Cell) -> float:
        # trade = self.step_count >= self.parameters['trading_start']
        cell_pos = TO_INDEX[cell.position]
        trade = False
        score = 0
        if cell.position in self.planned_moves:
            score -= 1500
            return score
        if cell.shipyard is not None:
            shipyard = cell.shipyard
            owner_id = shipyard.player_id
            if shipyard.player_id != self.player_id:
                if (
                    shipyard.player.halite < self.config.spawn_cost
                    and cell.ship is None
                    and ship.halite < 30
                    and sum(
                        [
                            1
                            for c in get_neighbours(cell)
                            if c.ship is not None
                            and c.ship.player_id == owner_id
                            and c.ship.halite <= ship.halite
                        ]
                    )
                    == 0
                ):  # The shipyard cannot be protected by the owner
                    score += 300
                elif ship.halite > self.parameters["max_halite_attack_shipyard"]:
                    score -= 400 + ship.halite
                elif (
                    ship.halite == 0
                    and (
                        (self.rank == 0 and self.ship_advantage > 0)
                        or self.step_count >= self.parameters["end_start"]
                        or cell_pos in self.farming_positions
                    )
                    or self.shipyard_distances[cell_pos] <= 2
                ):
                    score += 400  # Attack the enemy shipyard
                else:
                    score -= 300
            elif (
                self.halite >= self.spawn_cost
                and self.shipyard_count == 1
                and not self.spawn_limit_reached
            ):
                if (
                    self.step_count <= 100
                    or self.medium_dominance_map[TO_INDEX[shipyard.position]]
                    >= self.parameters["spawn_min_dominance"]
                ):
                    score += self.parameters["move_preference_block_shipyard"]
        if cell.ship is not None and cell.ship.player_id != self.player_id:
            if cell.ship.halite < ship.halite:
                score -= 750 + ship.halite - 0.5 * cell.ship.halite
            elif cell.ship.halite == ship.halite:
                if (
                    (
                        cell_pos not in self.farming_positions
                        or (
                            not trade
                            and cell.shipyard is None
                            and (
                                cell.ship.id not in self.intrusion_positions[cell_pos]
                                or self.intrusion_positions[cell_pos][cell.ship.id]
                                <= self.parameters["max_intrusion_count"]
                            )
                        )
                    )
                    and self.shipyard_distances[cell_pos] > 1
                    and (
                        self.next_shipyard_position is None
                        or get_distance(cell_pos, self.next_shipyard_position) > 2
                    )
                ):
                    score -= 450
            else:
                score += min(
                    cell.ship.halite * self.parameters["cell_score_enemy_halite"], 35
                )
        neighbour_value = 0
        for neighbour in get_neighbours(cell):
            if (
                neighbour.ship is not None
                and neighbour.ship.player_id != self.player_id
            ):
                if (
                    neighbour.ship.halite < ship.halite
                ):  # We really don't want to go to that cell unless it's necessary.
                    neighbour_value = -(750 + ship.halite) * (
                        self.parameters["cell_score_neighbour_discount"] + 0.15
                    )
                    break
                elif (
                    neighbour.ship.halite == ship.halite
                    and cell_pos not in self.shipyard_positions
                ):
                    if (
                        cell_pos not in self.farming_positions
                        and self.shipyard_distances[cell_pos] > 1
                        and self.shipyard_distances[TO_INDEX[neighbour.position]] > 1
                        and (
                            self.next_shipyard_position is None
                            or get_distance(cell_pos, self.next_shipyard_position) > 2
                        )
                    ) or (
                        not trade
                        and neighbour.shipyard is None
                        and (
                            neighbour.ship.id
                            not in self.intrusion_positions[
                                TO_INDEX[neighbour.position]
                            ]
                            or self.intrusion_positions[TO_INDEX[neighbour.position]][
                                neighbour.ship.id
                            ]
                            <= self.parameters["max_intrusion_count"]
                        )
                    ):
                        if (
                            self.step_count > self.parameters["greed_stop"]
                            or self.map_presence_diff[neighbour.ship.player_id]
                            >= self.parameters["greed_min_map_diff"]
                        ):  # TODO: check whether this is good
                            neighbour_value -= (
                                450 * self.parameters["cell_score_neighbour_discount"]
                            )
                else:
                    neighbour_value += min(
                        neighbour.ship.halite
                        * self.parameters["cell_score_enemy_halite"]
                        * self.parameters["cell_score_neighbour_discount"],
                        25,
                    )
        score += neighbour_value
        score += (
            self.parameters["cell_score_dominance"] * self.small_dominance_map[cell_pos]
        )
        if (
            cell_pos in self.farming_positions
            and 0 < cell.halite < self.harvest_threshold
        ):
            if TO_INDEX[ship.position] == cell_pos:
                score += self.parameters["cell_score_mine_farming"]
            else:
                score += self.parameters["cell_score_farming"]
        return score * (1 + self.parameters["cell_score_ship_halite"] * ship.halite)

    def calculate_player_score(self, player):
        return (
            player.halite
            + len(player.ships) * 500 * (1 - self.step_count / 398)
            + len(player.shipyards) * 750 * (1 - self.step_count / 398)
            + sum(
                [ship.halite / 4 for ship in player.ships]
                if len(player.ships) > 0
                else [0]
            )
        )

    def calculate_spawning_score(self, shipyard_position: int):
        if self.step_count <= self.parameters["farming_start"]:
            return self.ultra_blurred_halite_map[shipyard_position]
        dominance = self.medium_dominance_map[shipyard_position]
        if dominance < self.parameters["shipyard_abandon_dominance"]:
            return -999
        return -dominance

    def calculate_border_score(self, ship_pos, border_pos):
        score = (
            0.5 * get_distance(ship_pos, border_pos)
            + self.small_dominance_map[border_pos]
            + (
                50
                if ship_pos == border_pos and self.observation["halite"][border_pos] > 0
                else 0
            )
        )
        if self.shipyard_distances[border_pos] <= 2:
            score -= 1.5
        return score

    def calculate_player_map_presence(self, player):
        return len(player.ships) + len(player.shipyards)

    def estimate_farming_end(self):
        farming_positions = [TO_INDEX[point] for point in self.real_farming_points]
        avg_halite = sum(
            [0] + [self.observation["halite"][pos] for pos in farming_positions]
        ) / max(len(farming_positions), 1)
        avg_return_distance = sum(
            [0] + [self.shipyard_distances[pos] for pos in farming_positions]
        ) / max(len(farming_positions), 1)
        avg_ship_distance = sum(
            [0]
            + [
                self.shipyard_distances[TO_INDEX[ship.position]]
                for ship in self.me.ships
            ]
        ) / max(len(self.me.ships), 1)
        ship_count = int(len(self.me.ships) * 0.6)  # conservative estimate
        positions_per_ship = len(farming_positions) / max(ship_count, 1)
        steps_per_position = ceil(math.log((10 / max(avg_halite, 1)), 0.75))
        steps_per_ship = positions_per_ship * steps_per_position
        steps_between_positions = max(ceil(2 * positions_per_ship - 2), 0)
        steps_per_ship += (
            steps_between_positions
            + ceil(avg_ship_distance)
            + ceil(2.25 * avg_return_distance * max(positions_per_ship, 1))
            + self.parameters["end_return_extra_moves"]
            + 9
        )
        farming_end = clip(398 - ceil(steps_per_ship), 325, 365)
        return farming_end

    def prefer_moves(
        self,
        ship,
        directions,
        longest_axis,
        weight,
        penalize_farming=True,
        reduce_farming_penalty=False,
        destination=None,
    ):
        for dir in directions:
            position = (ship.position + dir.to_point()) % self.size
            w = weight
            if dir in longest_axis:
                w += self.parameters["move_preference_longest_axis"]
            self.change_position_score(ship, position, weight)
        if (
            destination is not None
            and len(directions) >= 2
            and len(self.real_farming_points) > 0
        ):
            axis1_farming_positions = self.get_farming_positions_count_in_between(
                ship.position, destination, get_axis(directions[0])
            )
            axis2_farming_positions = self.get_farming_positions_count_in_between(
                ship.position, destination, get_axis(directions[1])
            )
            if axis1_farming_positions > axis2_farming_positions:
                position = (ship.position + directions[0].to_point()) % self.size
                self.change_position_score(ship, position, int(-weight // 2))
            elif axis1_farming_positions < axis2_farming_positions:
                position = (ship.position + directions[1].to_point()) % self.size
                self.change_position_score(ship, position, int(-weight // 2))
        for dir in get_inefficient_directions(directions):
            position = (ship.position + dir.to_point()) % self.size
            self.change_position_score(ship, position, int(-weight // 1.2))
        if not penalize_farming:
            for cell in get_neighbours(ship.cell) + [ship.cell]:
                if (
                    TO_INDEX[cell.position] in self.farming_positions
                    and 0 < cell.halite < self.harvest_threshold
                ):
                    self.change_position_score(
                        ship, cell.position, -self.parameters["cell_score_farming"]
                    )
        elif reduce_farming_penalty:
            for cell in get_neighbours(ship.cell) + [ship.cell]:
                if (
                    TO_INDEX[cell.position] in self.farming_positions
                    and 0 < cell.halite < self.harvest_threshold
                ):
                    self.change_position_score(
                        ship,
                        cell.position,
                        int(-self.parameters["cell_score_farming"] // 1.5),
                    )

    def get_farming_positions_count_in_between(self, source, destination, axis):
        count = 0
        source_coordinate1 = source.x if axis == "x" else source.y
        source_coordinate2 = source.y if axis == "x" else source.x
        destination_coordinate = destination.y if axis == "x" else destination.x
        distance = dist(source_coordinate2, destination_coordinate)
        for farming_position in self.real_farming_points:
            farming_coordinate1 = (
                farming_position.x if axis == "x" else farming_position.y
            )
            if source_coordinate1 != farming_coordinate1:
                continue
            farming_coordinate2 = (
                farming_position.y if axis == "x" else farming_position.x
            )
            if (
                dist(source_coordinate2, farming_coordinate2) < distance
                and dist(destination_coordinate, farming_coordinate2) < distance
            ):
                count += 1
        return count

    def change_position_score(self, ship: Ship, position: Point, delta: float):
        self.ship_position_preferences[
            self.ship_to_index[ship], self.position_to_index[position]
        ] += delta

    def get_nearest_shipyard(self, pos: Point):
        min_distance = float("inf")
        nearest_shipyard = None
        for shipyard in self.me.shipyards:
            distance = calculate_distance(pos, shipyard.position)
            if distance < min_distance:
                min_distance = distance
                nearest_shipyard = shipyard
        return nearest_shipyard

    def get_friendly_neighbour_count(self, cell: Cell):
        return sum(
            1
            for _ in filter(
                lambda n: n.ship is not None and n.ship.player_id == self.player_id,
                get_neighbours(cell),
            )
        )

    def convert_to_shipyard(self, ship: Ship):
        assert self.halite + ship.halite >= self.config.convert_cost
        # ship.next_action = ShipAction.CONVERT
        self.ship_types[ship.id] = ShipType.CONVERTING
        self.ship_position_preferences[
            self.ship_to_index[ship], len(self.position_to_index) :
        ] = 9999999  # TODO: fix the amount of available shipyard conversions
        self.halite += ship.halite
        self.halite -= self.config.convert_cost
        self.ship_count -= 1
        self.shipyard_count += 1
        self.planned_shipyards.append(ship.position)
        if self.step_count < 10:
            self.first_shipyard_step = self.step_count
        if TO_INDEX[ship.position] == self.next_shipyard_position:
            self.next_shipyard_position = None

    def spawn_ship(self, shipyard: Shipyard):
        assert self.halite >= self.config.spawn_cost
        shipyard.next_action = ShipyardAction.SPAWN
        self.planned_moves.append(shipyard.position)
        self.halite -= self.config.spawn_cost
        self.ship_count += 1
        for cell in [shipyard.cell] + get_neighbours(shipyard.cell):
            if cell.ship is not None and cell.ship.player_id == self.player_id:
                self.change_position_score(cell.ship, shipyard.position, -1500)
        logging.debug(
            "Spawning ship on position "
            + str(shipyard.position)
            + " (shipyard "
            + str(shipyard.id)
            + ")"
        )

    def attack_position(self, position: Point):
        self.ship_position_preferences[:, self.position_to_index[position]][
            self.ship_position_preferences[:, self.position_to_index[position]] > -50
        ] += 900

    def calculate_harvest_threshold(self):
        ships_farming_points = len(self.me.ships) / max(
            len(self.real_farming_points), 1
        )
        threshold = clip(1 * self.step_count + 70, 80, 480)
        ship_advantage = (
            self.parameters["harvest_threshold_beta"]
            * clip(
                self.ship_advantage
                + self.parameters["harvest_threshold_ship_advantage_norm"],
                0,
                1.5 * self.parameters["harvest_threshold_ship_advantage_norm"],
            )
            / self.parameters["harvest_threshold_ship_advantage_norm"]
        )  # 0 <= this <= beta * 1.5
        threshold *= (
            1
            - (self.parameters["harvest_threshold_alpha"] / 2)
            + (
                self.parameters["harvest_threshold_alpha"]
                * (
                    1
                    - clip(
                        self.enemy_hunting_proportion,
                        0,
                        self.parameters["harvest_threshold_hunting_norm"],
                    )
                    / self.parameters["harvest_threshold_hunting_norm"]
                )
            )
        ) * (1 - (2 * self.parameters["harvest_threshold_beta"] / 3) + ship_advantage)
        threshold = 0.95 * threshold + 0.1 * clip(
            (ships_farming_points - 0.9) / 1.1, 0, 1
        )
        return int(clip(threshold, 110, 450))


def agent(obs, config):
    global BOT
    if BOT is None:
        BOT = HaliteBot(PARAMETERS)
    board = Board(obs, config)
    # logging.debug("Begin step " + str(board.step))
    return BOT.step(board, obs)