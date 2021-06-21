import src.helper as helper
import src.conf as conf
import numpy as np


def obs_to_matrix_baseline(obs, control_ship_id, player_id, own_ship_actions, size):
    ''' Baseline input function. Takes into consideration the following separate 9 images of size 21x21:
    halite, ships, shipyards, enemy ships, enemy shipyards, enemy ship cargos, enemy shipyards, 
    current ship cargo, player halite and timeframe. '''

    own_ships = obs['players'][player_id][2]
    control_ship_data = own_ships[control_ship_id]

    # Board images
    halite_list = [x/500 for x in obs['halite']]
    own_ships_list = [0] * (size*size)
    own_ships_unknown_list = [0] * (size*size)
    shipyard_list = [0] * (size*size)
    enemy_list = [0] * (size*size)
    enemy_small_list = [0] * (size*size)

    # Single digit filters
    timeframe = [obs['step']/conf.get('episodeSteps')] * (size*size)
    ship_cargo = [own_ships[control_ship_id][1]/1000] * (size*size)

    for idx, player in enumerate(obs['players']):
        ships = player[2]
        shipyards = player[1]
        for ship_id, ship_data in ships.items():
            if ship_id != control_ship_id:
                if idx != player_id:
                    if ship_data[1] > control_ship_data[1]:
                        ship_poses = [helper.get_next_ship_pos(ship_data[0], i) for i in range(5)]
                        for ship_pos in ship_poses:
                            enemy_small_list[ship_pos] = 1
                    else:
                        ship_poses = [helper.get_next_ship_pos(ship_data[0], i) for i in range(5)]
                        for ship_pos in ship_poses:
                            enemy_list[ship_pos] = 0.5

                elif idx == player_id:
                    if ship_id in own_ship_actions:
                        ship_pos = helper.get_next_ship_pos(ship_data[0], own_ship_actions[ship_id])
                        own_ships_list[ship_pos] = 1
                    else:
                        own_ships_unknown_list[ship_data[0]] = 1

        for shipyard_id, shipyard_pos in shipyards.items():
            if idx == player_id:
                shipyard_list[shipyard_pos] = 1
            else:
                enemy_list[shipyard_pos] = 1

    ship_pos = own_ships[control_ship_id][0]

    result = []
    for parts in zip(helper.center_cell(halite_list, ship_pos), helper.center_cell(own_ships_list, ship_pos), helper.center_cell(own_ships_unknown_list, ship_pos), helper.center_cell(shipyard_list, ship_pos), helper.center_cell(enemy_list, ship_pos), helper.center_cell(enemy_small_list, ship_pos), timeframe, ship_cargo):
        result.append(list(parts))

    image = np.reshape(result, [1, size, size, 8])

    # data = np.array([ship_cargo, timeframe])
    # data = np.reshape(data, [1, 2])

    mid = int(size/2)
    shipyards = helper.center_cell(shipyard_list, ship_pos)
    nearby_shipyards = np.reshape(shipyards, [1, size, size, 1])[:, mid-7:mid+8, mid-7:mid+8, :]
    has_nearby_shipyards = nearby_shipyards.sum() != 0

    return image, has_nearby_shipyards


def obs_to_matrix_six(obs, control_ship_id, player_id, own_ship_actions, size):
    ''' Uses only 4 images and 2 single color filters '''
    result = []
    own_ships = obs['players'][player_id][2]
    control_ship_data = own_ships[control_ship_id]

    # Multi value maps
    halite_list_and_enemies = [x/500 for x in obs['halite']]
    shipyard_list = [0] * (size*size)
    own_ships_list = [0] * (size*size)
    enemy_list = [0] * (size*size)

    # Single value filters
    timeframe = [obs['step']/conf.get('episodeSteps')] * (size*size)
    ship_cargo = [own_ships[control_ship_id][1]/1000] * (size*size)
    for idx, player in enumerate(obs['players']):
        ships = player[2]
        shipyards = player[1]
        for ship_id, ship_data in ships.items():
            if ship_id != control_ship_id:
                if idx != player_id:
                    if ship_data[1] > control_ship_data[1]:

                        # Setting halite and big enemy cargos
                        halite_list_and_enemies[ship_data[0]] += ship_data[1]/500
                    else:
                        ship_poses = [helper.get_next_ship_pos(ship_data[0], i) for i in range(5)]
                        for ship_pos in ship_poses:
                            # Setting dangers
                            enemy_list[ship_pos] = 0.5

                elif idx == player_id and ship_id in own_ship_actions:
                    # Setting friendly ships
                    ship_pos = helper.get_next_ship_pos(ship_data[0], own_ship_actions[ship_id])
                    own_ships_list[ship_pos] = 1

        
        for shipyard_id, shipyard_pos in shipyards.items():
            if idx == player_id:
                # Setting friendly shipyards
                shipyard_list[shipyard_pos] = 1
            else:
                # Setting enemy shipyards
                enemy_list[shipyard_pos] = 1

    ship_pos = own_ships[control_ship_id][0]

    for parts in zip(helper.center_cell(halite_list_and_enemies, ship_pos), helper.center_cell(shipyard_list, ship_pos), helper.center_cell(own_ships_list, ship_pos), helper.center_cell(enemy_list, ship_pos), timeframe, ship_cargo):
        result.append(list(parts))

    image = np.reshape(result, [1, size, size, 6])
    
    mid = int(size/2)
    shipyards = helper.center_cell(shipyard_list, ship_pos)
    nearby_shipyards = np.reshape(shipyards, [1, size, size, 1])[:, mid-7:mid+8, mid-7:mid+8, :]
    has_nearby_shipyards = nearby_shipyards.sum() != 0

    return image, has_nearby_shipyards


def obs_to_matrix_sparse(obs, control_ship_id, player_id, own_ship_actions, size):
    ''' Baseline input function. Takes into consideration the following separate 9 images of size 21x21:
    halite, ships, shipyards, enemy ships, enemy shipyards, enemy ship cargos, enemy shipyards, 
    current ship cargo, player halite and timeframe. '''

    own_ships = obs['players'][player_id][2]
    control_ship_data = own_ships[control_ship_id]

    enemy_id = 1 if player_id == 0 else 0

    # Board images
    halite_list = [x/500 for x in obs['halite']]
    own_ships_list = [0] * (size*size)
    own_ships_unknown_list = [0] * (size*size)
    shipyard_list = [0] * (size*size)
    enemy_list = [0] * (size*size)
    enemy_small_list = [0] * (size*size)

    # Single digit filters
    timeframe = [obs['step']/conf.get('episodeSteps')] * (size*size)
    ship_cargo = [own_ships[control_ship_id][1]/1000] * (size*size)
    halite = [obs['players'][player_id][0] / (obs['players'][player_id][0] + obs['players'][enemy_id][0] + 1)] * (size*size)

    for idx, player in enumerate(obs['players']):
        ships = player[2]
        shipyards = player[1]
        for ship_id, ship_data in ships.items():
            if ship_id != control_ship_id:
                if idx != player_id:
                    if ship_data[1] > control_ship_data[1]:
                        ship_poses = [helper.get_next_ship_pos(ship_data[0], i) for i in range(5)]
                        for ship_pos in ship_poses:
                            enemy_small_list[ship_pos] = 1
                    else:
                        ship_poses = [helper.get_next_ship_pos(ship_data[0], i) for i in range(5)]
                        for ship_pos in ship_poses:
                            enemy_list[ship_pos] = 0.5

                elif idx == player_id:
                    if ship_id in own_ship_actions:
                        ship_pos = helper.get_next_ship_pos(ship_data[0], own_ship_actions[ship_id])
                        own_ships_list[ship_pos] = 1
                    else:
                        own_ships_unknown_list[ship_data[0]] = 1

        for shipyard_id, shipyard_pos in shipyards.items():
            if idx == player_id:
                shipyard_list[shipyard_pos] = 1
            else:
                enemy_list[shipyard_pos] = 1

    ship_pos = own_ships[control_ship_id][0]


    result = []
    for parts in zip(helper.center_cell(halite_list, ship_pos), helper.center_cell(own_ships_list, ship_pos), helper.center_cell(own_ships_unknown_list, ship_pos), helper.center_cell(shipyard_list, ship_pos), helper.center_cell(enemy_list, ship_pos), helper.center_cell(enemy_small_list, ship_pos), timeframe, ship_cargo, halite):
        result.append(list(parts))

    image = np.reshape(result, [1, size, size, 9])

    # data = np.array([ship_cargo, timeframe])
    # data = np.reshape(data, [1, 2])

    mid = int(size/2)
    shipyards = helper.center_cell(shipyard_list, ship_pos)
    nearby_shipyards = np.reshape(shipyards, [1, size, size, 1])[:, mid-7:mid+8, mid-7:mid+8, :]
    has_nearby_shipyards = nearby_shipyards.sum() != 0

    return image, has_nearby_shipyards

def obs_to_matrix_nearsight(obs, control_ship_id, player_id, own_ship_actions, size):
    ''' Baseline input function. Takes into consideration the following separate 9 images of size 21x21:
    halite, ships, shipyards, enemy ships, enemy shipyards, enemy ship cargos, enemy shipyards, 
    current ship cargo, player halite and timeframe. '''

    own_ships = obs['players'][player_id][2]
    control_ship_data = own_ships[control_ship_id]

    # Board images
    halite_list = [x/500 for x in obs['halite']]
    own_ships_list = [0] * (size*size)
    own_ships_unknown_list = [0] * (size*size)
    shipyard_list = [0] * (size*size)
    enemy_list = [0] * (size*size)
    enemy_small_list = [0] * (size*size)

    # Single digit filters
    timeframe = [obs['step']/conf.get('episodeSteps')] * (size*size)
    ship_cargo = [own_ships[control_ship_id][1]/1000] * (size*size)

    for idx, player in enumerate(obs['players']):
        ships = player[2]
        shipyards = player[1]
        for ship_id, ship_data in ships.items():
            if ship_id != control_ship_id:
                if idx != player_id:
                    if ship_data[1] > control_ship_data[1]:
                        ship_poses = [helper.get_next_ship_pos(ship_data[0], i) for i in range(5)]
                        for ship_pos in ship_poses:
                            enemy_small_list[ship_pos] = 1
                    else:
                        ship_poses = [helper.get_next_ship_pos(ship_data[0], i) for i in range(5)]
                        for ship_pos in ship_poses:
                            enemy_list[ship_pos] = 0.5

                elif idx == player_id:
                    if ship_id in own_ship_actions:
                        ship_pos = helper.get_next_ship_pos(ship_data[0], own_ship_actions[ship_id])
                        own_ships_list[ship_pos] = 1
                    else:
                        own_ships_unknown_list[ship_data[0]] = 1

        for shipyard_id, shipyard_pos in shipyards.items():
            if idx == player_id:
                shipyard_list[shipyard_pos] = 1
            else:
                enemy_list[shipyard_pos] = 1

    ship_pos = own_ships[control_ship_id][0]

    # lists = [halite_list, own_ships_list, own_ships_unknown_list, shipyard_list, enemy_list, enemy_small_list, timeframe, ship_cargo]
    # zippable = [helper.center_cell(a, ship_pos) for a in lists]
    result = []
    for parts in zip(helper.center_cell(halite_list, ship_pos), helper.center_cell(own_ships_list, ship_pos), helper.center_cell(own_ships_unknown_list, ship_pos), helper.center_cell(shipyard_list, ship_pos), helper.center_cell(enemy_list, ship_pos), helper.center_cell(enemy_small_list, ship_pos), timeframe, ship_cargo):
        result.append(list(parts))

    # for parts in zip(zippable):
    #     result.append(list(parts))

    image = np.reshape(result, [1, size, size, 8])

    # data = np.array([ship_cargo, timeframe])
    # data = np.reshape(data, [1, 2])

    mid = int(size/2)
    image = image[:, mid-5:mid+6, mid-5:mid+6, :]
    shipyards = helper.center_cell(shipyard_list, ship_pos)
    nearby_shipyards = np.reshape(shipyards, [1, size, size, 1])[:, mid-7:mid+8, mid-7:mid+8, :]
    has_nearby_shipyards = nearby_shipyards.sum() != 0

    return image, has_nearby_shipyards

def obs_to_matrix_three(obs, control_ship_id, player_id, own_ship_actions, size):
    ''' Uses only 1 image and 2 single color filters '''
    result = []
    own_ships = obs['players'][player_id][2]
    control_ship_data = own_ships[control_ship_id]

    # Multi value maps
    full_list = [x/500 for x in obs['halite']]

    # Single value filters
    timeframe = [obs['step']/conf.get('episodeSteps')] * (size*size)
    ship_cargo = [own_ships[control_ship_id][1]/1000] * (size*size)
    for idx, player in enumerate(obs['players']):
        ships = player[2]
        shipyards = player[1]
        for ship_id, ship_data in ships.items():
            if ship_id != control_ship_id:
                if idx != player_id:
                    if ship_data[1] > control_ship_data[1]:
                        # Setting halite and big enemy cargos
                        full_list[ship_data[0]] += ship_data[1]/500
                    else:
                        ship_poses = [helper.get_next_ship_pos(ship_data[0], i) for i in range(5)]
                        for ship_pos in ship_poses:
                            # Setting dangers
                            full_list[ship_pos] = -1

                elif idx == player_id and ship_id in own_ship_actions:
                    # Setting friendly ships
                    ship_pos = helper.get_next_ship_pos(ship_data[0], own_ship_actions[ship_id])
                    full_list[ship_pos] = -1

        
        for shipyard_id, shipyard_pos in shipyards.items():
            # Setting enemy shipyards
            full_list[shipyard_pos] = -1

    ship_pos = own_ships[control_ship_id][0]

    for parts in zip(helper.center_cell(full_list, ship_pos)):
        result.append(list(parts))

    image = np.reshape(result, [1, size, size, 3])
    
    mid = int(size/2)
    shipyards = helper.center_cell(shipyard_list, ship_pos)
    nearby_shipyards = np.reshape(shipyards, [1, size, size, 1])[:, mid-7:mid+8, mid-7:mid+8, :]
    has_nearby_shipyards = nearby_shipyards.sum() != 0

    return image, has_nearby_shipyards