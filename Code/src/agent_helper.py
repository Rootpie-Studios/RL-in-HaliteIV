from kaggle_environments.envs.halite.helpers import *

import numpy as np
import random

def get_action(epsilon, ship_agent, state):
    Qs = ship_agent(state, training=False)[0]
    # Explore
    if epsilon > np.random.rand():
        action = np.random.choice(5)
    # Exploit
    else:
        action = np.argmax(Qs)
        
    return action, Qs

def convert_strat(me, ship, has_nearby_shipyards, action, shipyard_build):
    if me.halite + ship.halite > 500 and not has_nearby_shipyards and not shipyard_build:
        ship.next_action = ShipAction.CONVERT
        action = 5
        shipyard_build = True

    return action, shipyard_build

def convert_override(me, sorted_ships):
    if len(me.shipyards) == 0 and len(me.ships) > 0:
        ship = sorted_ships[0]
        if ship.halite + me.halite > 500 and len(me.ships) > 1:
            ship.next_action = ShipAction.CONVERT
        elif ship.halite + me.halite > 1000:
            ship.next_action = ShipAction.CONVERT

def get_epsilon(epsilon_default, epsilon_decay, epsilon_decay_rate, epsilon_min, nbr_games):
    # Calculate epsilon for exploration rate
    if epsilon_decay:
        epsilon = epsilon_default - epsilon_decay_rate * nbr_games
        epsilon = max(epsilon_min, epsilon)
    else:
        epsilon = epsilon_default

    return epsilon
    
def shipyard_strat(me, obs, board, new_ship_pos, episode_steps):
    if len(me.shipyards) > 0:
        shipyard = random.choice(list(me.shipyards))
        ship_nearby = False

        if obs['players'][me._id][1][shipyard._id] in new_ship_pos:
            ship_nearby = True

        if me._halite >= 500 and not ship_nearby and board.step < episode_steps - 200 and len(me.ships) < 25:
            shipyard.next_action = ShipyardAction.SPAWN
        elif me._halite >= 500 and not ship_nearby and len(me.ships) < 10 and board.step < episode_steps - 10:
            shipyard.next_action = ShipyardAction.SPAWN