from kaggle_environments.envs.halite.helpers import *
from kaggle_environments import make
import numpy as np
import math, os, pickle, time, re

def play_halite(agent1, agent2, folder, html_folder, configuration):
    ''' Plays halite and saves games and html_games to folders set in conf.py '''
    gameID = str(len(os.listdir(folder)))
    configuration.update({'randomSeed' : int(gameID)})
    env = make('halite', debug=True, configuration=configuration)
    start = time.time()
    obs = env.run([agent1, agent2])
    print('Game time:', time.time() - start)
    last_obs = obs[-1][0]

    # Pickle saving
    with open(folder + '/' + gameID, 'rb') as pickle_file:
        memory = pickle.load(pickle_file)

    memory.update({'players' : last_obs['observation']['players']})
    player_won(last_obs['observation']['players'], alive(last_obs['observation']['players']))

    with open(folder + '/' + gameID, 'wb') as pickle_file:
        pickle.dump(memory, pickle_file)

    # HTML saving
    if int(gameID) % 25 == 0:
        with open(html_folder + '/' + gameID + '.html', 'wt') as fd:
            fd.write(env.render(mode='html'))

    print('Game saved\n###########')

def play_halite_sparse(agent1, agent2, folder, html_folder, configuration):
    ''' Plays halite and saves games and html_games to folders set in conf.py '''
    gameID = str(len(os.listdir(folder)))
    configuration.update({'randomSeed' : int(gameID)})
    env = make('halite', debug=True, configuration=configuration)
    start = time.time()
    obs = env.run([agent1, agent2])
    print('Game time:', time.time() - start, ' sparse vs ' + agent2[7:-3])
    last_obs = obs[-1][0]

    # Pickle saving
    with open(folder + '/' + gameID, 'rb') as pickle_file:
        memory = pickle.load(pickle_file)

    memory.update({'players' : last_obs['observation']['players']})
    player_won(last_obs['observation']['players'], alive(last_obs['observation']['players']))

    with open(folder + '/' + gameID, 'wb') as pickle_file:
        pickle.dump(memory, pickle_file)

    # HTML saving
    with open(html_folder + '/sparse' + '_vs_' + agent2[7:-3] + '.html', 'wt') as fd:
        fd.write(env.render(mode='html'))

    print('Game saved\n###########')

def ship_action_index(action_string):
    ''' Takes input action string and returns number representation '''

    if (action_string == 'EAST'):
        return 1
    elif (action_string == 'NORTH'):
        return 2
    elif (action_string == 'SOUTH'):
        return 3
    elif (action_string == 'WEST'):
        return 4
    else: #CONVERT
        return 5

def ship_action_string(action_index):
    ''' Takes input action string and returns number representation '''

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
    ''' Takes input_list (=one image representation of data from game 21x21) and returns it as
    an equivalent representation with input center_pos as center cell '''

    size = int(math.sqrt(len(input_list)))
    half_size = math.floor(size/2)

    mat = np.reshape(input_list, (size, size))
    res_mat = np.zeros([size*2, size*2])

    res_mat[0:0+mat.shape[0], 0:0+mat.shape[1]] += mat
    res_mat[size:size+mat.shape[0], 0:0+mat.shape[1]] += mat
    res_mat[0:0+mat.shape[0], size:size+mat.shape[1]] += mat
    res_mat[size:size+mat.shape[0], size:size+mat.shape[1]] += mat

    row = center_pos%size
    col = math.floor(center_pos/size)
    if row <= size/2 and col <= size/2:
        quad = (1, 1)
    elif row <= size/2 and col >= size/2:
        quad = (1, 0)
    elif row >= size/2 and col <= size/2:
        quad = (0, 1)
    else:
        quad = (0, 0)

    row = row + quad[0]*size
    col = col + quad[1]*size

    res_mat = res_mat[col-half_size:col+half_size+1, row-half_size:row+half_size+1]

    return np.reshape(res_mat, (1, size*size))[0]

def alive(players):
    ''' Returns list representation on format [p1, p2] describing if p1 and p2 are alive (=1) '''
    alive = [1, 1]

    for player_index, player in enumerate(players):
        if len(player[2]) == 0 and player[0] < 500 or (len(player[2]) == 0 and len(player[1]) == 0):
            alive[player_index] = 0

    return alive

def player_won(players, alive):
    ''' Returns 0 if player 0 won, 1 otherwise '''
    if alive[0] == 1 and alive[1] == 1:
        return 0 if players[0][0] > players[1][0] else 1
    else:
        return 0 if alive[0] == 1 else 1

def get_player_tot_cargo(players, player_id):
    ''' Returns the sum of cargo of all ships for player '''
    tot_cargo = sum([x[1] for x in list(players[player_id][2].values())])
    return tot_cargo

def get_next_ship_pos(ship_pos, action):
    ''' Returns the next position from intended action '''
    next_ship_pos = -1
 
    if action == 0:
        next_ship_pos = ship_pos
    elif action == 1:
        next_ship_pos = ((ship_pos)//21)*21 + (ship_pos+1)%21
    elif action == 2:
        next_ship_pos = ((ship_pos-21)//21)*21 + (ship_pos)%21
    elif action == 3:
        next_ship_pos = (((ship_pos+21)//21)*21 + (ship_pos)%21) % 441
    elif action == 4:
        next_ship_pos = ((ship_pos)//21)*21 + (ship_pos-1)%21
 
    return next_ship_pos
 
def get_next_ship_poses(ships, actions, current_ship_id):
    ''' Returns a list of next positions from the intended action they intend to make '''
    next_ship_poses = []
 
    for ship_id, ship_data in ships.items():
        if ship_id in actions and ship_id != current_ship_id:
            ship_pos = get_next_ship_pos(ship_data[0], actions[ship_id])
 
            next_ship_poses.append(ship_pos)
 
    return next_ship_poses

def extend_matrix(mat, extend_size):
    ''' Returns an extended numpy matrix with elements on its opposite side '''
    return np.pad(mat, extend_size, mode='wrap')[0]

def atoi(text):
    ''' Converts string to int if it is made of digits '''
    return int(text) if text.isdigit() else text

def natural_keys(text):
    ''' Returns list of ints from string '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def get_danger_poses(obs, ship_info, enemy_id=1):
    ''' Returns a list of all poses with shipyards or ships with less cargo than input ship_info ship '''
    danger_poses = []
    for enemy_ship_id, enemy_ship_data in obs['players'][enemy_id][2].items():
        if enemy_ship_data[1] < ship_info[1]: 
            enemy_ship_possible_poses = [get_next_ship_pos(enemy_ship_data[0], i) for i in range(5)]
            danger_poses.extend(enemy_ship_possible_poses)

    for enemy_shipyard_id, enemy_shipyard_pos in obs['players'][enemy_id][1].items():
        danger_poses.append(enemy_shipyard_pos)

    return danger_poses

def get_kill_poses(obs, ship_info, enemy_id):
    kill_poses = []

    for enemy_ship_id, enemy_ship_data in obs['players'][enemy_id][2].items():
        if enemy_ship_data[1] >= ship_info[1]: 
            enemy_ship_possible_poses = [get_next_ship_pos(enemy_ship_data[0], i) for i in range(5)]
            kill_poses.extend(enemy_ship_possible_poses)

    for enemy_shipyard_id, enemy_shipyard_pos in obs['players'][enemy_id][1].items():
        kill_poses.append(enemy_shipyard_pos)

    return kill_poses
