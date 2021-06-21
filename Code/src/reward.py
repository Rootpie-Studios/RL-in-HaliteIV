import src.helper as helper
import numpy as np

def halite_reward(ns, ship_action, ship_cargo, ship_cargo_next, last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, step, file_name, own_estimation):
   os = False
   if ns:
       if next_ship_pos in danger_poses:
        #    print('SHIP COULD HAVE DIED... IN STEP: ', step, ' GAME: ', file_name)
           reward = -50
       elif next_ship_pos in next_ship_poses:
        #    print('KILLED OWN SHIP :( IN STEP: ', step, ' GAME: ', file_name)
           reward = -500
       elif ship_cargo_next > ship_cargo:
           reward = ship_cargo_next - ship_cargo
       elif ship_cargo > 0 and ship_cargo_next == 0:
           reward = ship_cargo
        #    print('SHIP RETURNED {} CARGO AT SHIPYARD IN STEP: '.format(ship_cargo), step, ' GAME: ', file_name)
       else:
           reward = -1
 
   else:
       if last_frame:
           reward = -ship_cargo
       else:
           if next_ship_pos in danger_poses or next_ship_pos in next_ship_poses:
            #    print('SHIP DIED BY IGNORANCE IN STEP: ', step, ' GAME: ', file_name)
               reward = - (ship_cargo + 500)
           else:
            #    print('OWN SHIP KILLED US BY FOR NO REASON IN STEP: ', step, ' GAME: ', file_name)
                reward = own_estimation
                os = True
 
   return reward, os

def final_reward(ns, ship_action, ship_cargo, ship_cargo_next, last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, step, file_name, own_estimation):
   if ns:
       if next_ship_pos in danger_poses:
        #    print('SHIP COULD HAVE DIED... IN STEP: ', step, ' GAME: ', file_name)
           reward = -50
       elif next_ship_pos in next_ship_poses:
        #    print('KILLED OWN SHIP :( IN STEP: ', step, ' GAME: ', file_name)
           reward = -5000
       elif ship_cargo_next > ship_cargo:
           reward = ship_cargo_next - ship_cargo
       elif ship_cargo > 0 and ship_cargo_next == 0:
           reward = ship_cargo
        #    print('SHIP RETURNED {} CARGO AT SHIPYARD IN STEP: '.format(ship_cargo), step, ' GAME: ', file_name)
       else:
           reward = -1
 
   else:
       if last_frame:
           reward = -ship_cargo
       else:
           if next_ship_pos in danger_poses or next_ship_pos in next_ship_poses:
            #    print('SHIP DIED BY IGNORANCE IN STEP: ', step, ' GAME: ', file_name)
               reward = - (ship_cargo + 500)
           else:
            #    print('OWN SHIP KILLED US BY FOR NO REASON IN STEP: ', step, ' GAME: ', file_name)
                reward = own_estimation
 
   return reward

def team_spirit(ns, ship_action, ship_cargo, ship_cargo_next, last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, step, file_name, own_estimation):
    if ns:
        if next_ship_pos in danger_poses:
            #    print('SHIP COULD HAVE DIED... IN STEP: ', step, ' GAME: ', file_name)
            reward = -500
        elif next_ship_pos in next_ship_poses:
            #    print('KILLED OWN SHIP :( IN STEP: ', step, ' GAME: ', file_name)
            reward = -10000 * ((400-step) / 400)
        elif ship_cargo_next > ship_cargo:
           reward = (ship_cargo_next - ship_cargo)/4
        elif ship_cargo > 0 and ship_cargo_next == 0:
            reward = ship_cargo
            #    print('SHIP RETURNED {} CARGO AT SHIPYARD IN STEP: '.format(ship_cargo), step, ' GAME: ', file_name)
        else:
            reward = -1
    
    else:
        if last_frame:
            reward = -ship_cargo
        else:
            if next_ship_pos in danger_poses or next_ship_pos in next_ship_poses:
                #    print('SHIP DIED BY IGNORANCE IN STEP: ', step, ' GAME: ', file_name)
                    reward = - (ship_cargo + 500)
            else:
                #    print('OWN SHIP KILLED US BY FOR NO REASON IN STEP: ', step, ' GAME: ', file_name)
                    reward = own_estimation
    
    return reward

def no_friend(ns, ship_action, ship_cargo, ship_cargo_next, last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, step, file_name, own_estimation):
   os = False
   if ns:
       if next_ship_pos in danger_poses:
        #    print('SHIP COULD HAVE DIED... IN STEP: ', step, ' GAME: ', file_name)
           reward = -50
       elif next_ship_pos in next_ship_poses:
        #    print('KILLED OWN SHIP :( IN STEP: ', step, ' GAME: ', file_name)
           reward = -5000
       elif ship_cargo_next > ship_cargo:
           reward = ship_cargo_next - ship_cargo
       elif ship_cargo > 0 and ship_cargo_next == 0:
           reward = ship_cargo
        #    print('SHIP RETURNED {} CARGO AT SHIPYARD IN STEP: '.format(ship_cargo), step, ' GAME: ', file_name)
       else:
           reward = -1
 
   else:
       if last_frame:
           reward = -ship_cargo
       else:
           if next_ship_pos in danger_poses or next_ship_pos in next_ship_poses:
            #    print('SHIP DIED BY IGNORANCE IN STEP: ', step, ' GAME: ', file_name)
               reward = - (ship_cargo + 500)
           else:
            #    print('OWN SHIP KILLED US BY FOR NO REASON IN STEP: ', step, ' GAME: ', file_name)
                reward = 0
                os = True
 
   return reward, os
   
def won_reward(alive, players):
    if alive[0] == 1 and alive[1] == 1:
        won = 1 if players[0][0] > players[1][0] else -1
    else:
        won = 1 if alive[0] == 1 else -1

    return won

def alive(players):
    ''' Returns list representation on format [p1, p2] describing if p1 and p2 are alive (=1) '''
    alive = [1, 1]

    for player_index, player in enumerate(players):
        if len(player[2]) == 0 and player[0] < 500 or (len(player[2]) == 0 and len(player[1]) == 0):
            alive[player_index] = 0

    return alive
        
def more_halite(halite_now, halite_before):
    if halite_now > halite_before:
        return 2
    return 0

def farmed_or_stole(obs, next_obs, ship_id):
    ships = obs['players'][0][2]
    cargo = ships[ship_id][1]
    next_ships = next_obs['players'][0][2]
    next_cargo = next_ships[ship_id][1]

    if next_cargo > cargo:
        return next_cargo - cargo
    return 0

def destroyed_ships(num_ships_now, num_ships_before):
    if num_ships_now < num_ships_before:
        return -1
    return 0 

def destroyed_enemy_cargo_reward(obs, next_obs, ship_id):
    ships = obs['players'][0][2]
    cargo = ships[ship_id][1]
    next_ships = next_obs['players'][0][2]
    next_cargo = next_ships[ship_id][1]

    enemy_ships = obs['players'][1][2]
    next_enemy_ships = next_obs['players'][1][2]

    enemy_ship_ids = list(enemy_ships.keys())
    next_enemy_ship_ids = list(next_enemy_ships.keys())
    destroyed_enemy_ship_ids = set(enemy_ship_ids) - set(next_enemy_ship_ids)
    destroyed_enemy_ship_cargos = [enemy_ships[idd][1] for idd in destroyed_enemy_ship_ids]

    current_reward = 0
    stolen_cargo = next_cargo - cargo
    if stolen_cargo in destroyed_enemy_ship_cargos:
        current_reward = 500
    
    return current_reward

def destroyed_enemy_ship(obs, next_obs, ship_id):
    ships = obs['players'][0][2]
    cargo = ships[ship_id][1]
    next_ships = next_obs['players'][0][2]
    next_cargo = next_ships[ship_id][1]

    enemy_ships = obs['players'][1][2]
    next_enemy_ships = next_obs['players'][1][2]

    enemy_ship_ids = list(enemy_ships.keys())
    next_enemy_ship_ids = list(next_enemy_ships.keys())
    destroyed_enemy_ship_ids = set(enemy_ship_ids) - set(next_enemy_ship_ids)
    destroyed_enemy_ship_cargos = [enemy_ships[idd][1] for idd in destroyed_enemy_ship_ids]

    stolen_cargo = next_cargo - cargo
    if stolen_cargo in destroyed_enemy_ship_cargos and stolen_cargo > 0:
        return True
    
    return False

def convert_reward(obs, ship_id):
    ships = obs['players'][0][2]
    cargo = ships[ship_id][1]
    if cargo >= 500:
        return cargo - 1000
    return -100

def returned_halite(obs, next_obs, ship_id):
    ships = obs['players'][0][2]
    cargo = ships[ship_id][1]
    next_ships = next_obs['players'][0][2]
    next_cargo = next_ships[ship_id][1]

    if cargo != 0 and next_cargo == 0:
        return next_cargo
    else:
        return 0

def slack_penalty(action, current_reward):
    ret = 0
    if current_reward < 0:
        ret -= 1
        if action == 0:
            ret -= 99

    return ret

def reward_function(model, obs, next_obs, ship_id, current_ship_view, next_ship_pos, next_ship_poses, file, frame_nbr):
    r = 0
    
    if next_obs != False:
        next_ships = next_obs['players'][0][2]

        if ship_id in next_ships:
            next_state = helper.obs_to_matrix_compact(next_obs, ship_id, 0, model.size, current_ship_view)
            
            current_reward = farmed_or_stole(obs, next_obs, ship_id)
            current_reward += returned_halite(obs, next_obs, ship_id)
            future_reward = np.amax(model.model(next_state, training=False)[0]) * model.gamma
            r = current_reward + future_reward

            # print('Game: {}, Frame: {}, Nbr ships: {}, Next ship pos: {}, Next ship poses: {}'.format(file, frame_nbr, len(obs['players'][0][2]), next_ship_pos, next_ship_poses))
            if next_ship_pos in next_ship_poses:
                # print('Killed own ship. Game: {}, Frame: {}'.format(file, frame_nbr))
                r -= 500

        else:
            r = -500

    return r

