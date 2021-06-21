# Imports helper functions
from kaggle_environments.envs.halite.helpers import *
import logging, sys
import time
import pprint
import random, pickle, os, collections
import src.conf as conf
import src.helper as helper
import numpy as np

one_step = {ShipAction.NORTH: Point(0,1), ShipAction.EAST: Point(1,0),
            ShipAction.SOUTH: Point(0,-1), ShipAction.WEST: Point(-1,0)}
two_steps = {ShipAction.NORTH: [Point(0,2), Point(1,1), Point(-1,1)],
             ShipAction.EAST:  [Point(2,0), Point(1,1), Point(1,-1)],
             ShipAction.SOUTH: [Point(0,-2), Point(1,-1), Point(-1,-1)],
             ShipAction.WEST:  [Point(-2,0), Point(-1,-1), Point(-1,1)]}
one_step_away = [Point(0,1), Point(1,0), Point(0,-1), Point(-1,0)]
two_steps_away = [Point(0,2), Point(2,0), Point(0,-2), Point(-2,0),
                  Point(1,1), Point(1,-1), Point(-1,-1), Point(-1,1)]
pos_halite = [Point(0,0), Point(0,1), Point(1,0), Point(0,-1), Point(-1,0),
              Point(0,2), Point(2,0), Point(0,-2), Point(-2,0), Point(1,1), Point(1,-1), Point(-1,-1), Point(-1,1),
              Point(0,3), Point(3,0), Point(0,-3), Point(-3,0), Point(1,2), Point(1,-2), Point(-1,-2), Point(-1,2),
              Point(2,1), Point(2,-1), Point(-2,-1), Point(-2,1),
              Point(0,4), Point(4,0), Point(0,-4), Point(-4,0), Point(1,3), Point(1,-3), Point(-1,-3), Point(-1,3),
              Point(3,1), Point(3,-1), Point(-3,-1), Point(-3,1), Point(2,2), Point(2,-2), Point(-2,-2), Point(-2,2)]
pos_enemy =  [Point(0,0), Point(0,1), Point(1,0), Point(0,-1), Point(-1,0),
              Point(0,2), Point(2,0), Point(0,-2), Point(-2,0), Point(1,1), Point(1,-1), Point(-1,-1), Point(-1,1),
              Point(0,3), Point(3,0), Point(0,-3), Point(-3,0), Point(1,2), Point(1,-2), Point(-1,-2), Point(-1,2),
              Point(2,1), Point(2,-1), Point(-2,-1), Point(-2,1)]
directions = [ShipAction.NORTH, ShipAction.SOUTH, ShipAction.EAST, ShipAction.WEST]
size = 0
risk = {}
risksum = {}
under_attack = {}
ship_states = {}
ship_target = {}
shipyard_states = {}
s_env = {}
next_pos = []
new_yard_create = False

def our_ship(ship):
    if ship:
        if ship.player_id == board.current_player_id:
            return True
    return False

def our_shipyard(sy):
    if sy:
        if sy.player_id == board.current_player_id:
            return True
    return False

def ship_by_id(id):
    for ship in board.current_player.ships:
        if ship.id == id:
            return ship
    debugprint(f'Cant find ship id {id}')
    return None

def enemy_steps_from_shipyard(shipyard):
    for pos in one_step_away:
        npos = shipyard.position+pos
        nship = board[npos].ship
        if nship is not None and nship.player_id is not board.current_player_id:
            return 1
    for pos in two_steps_away:
        npos = shipyard.position+pos
        nship = board[npos].ship
        if nship is not None and nship.player_id is not board.current_player_id:
            return 2
    return 0

def find_steps_between_pos(fromPos, toPos):
    fromX, fromY = divmod(fromPos[0],size), divmod(fromPos[1],size)
    toX, toY = divmod(toPos[0],size), divmod(toPos[1],size)
    deltaX = toX[1]-fromX[1]
    deltaY = toY[1]-fromY[1]
    if abs(deltaX) <= size//2:
        stepX = abs(deltaX)
    else:
        if deltaX > 0:
            stepX = abs(deltaX - size)
        elif deltaX < 0:
            stepX = abs(deltaX + size)
    if abs(deltaY) <= size//2:
        stepY = abs(deltaY)
    else:
        if deltaY > 0:
            stepY = abs(deltaY - size)
        elif deltaY < 0:
            stepY = abs(deltaY + size)

    # debugprint(f'  Find steps from ({fromX[1]}, {fromY[1]}) to ({toX[1]}, {toY[1]}) = {stepX} + {stepY}')
    return stepX+stepY

def find_nearest_shipyard(fromPos):
    target_shipyard = None
    shortest_steps = size*2
    
    for shipyard in board.current_player.shipyards:
        steps = find_steps_between_pos(fromPos, shipyard.cell.position)
        if (steps < shortest_steps):
            shortest_steps = steps
            target_shipyard = shipyard
    return target_shipyard, shortest_steps

def next_cell_in_direction (fromX, fromY, direction):
    x_offset = 0
    y_offset = 0
    if direction == ShipAction.NORTH:
        y_offset = 1
        # debugprint (f'  N y_offset {y_offset}')
    elif direction == ShipAction.SOUTH:
        y_offset = -1
        # debugprint (f'  S y_offset {y_offset}')
    elif direction == ShipAction.EAST:
        x_offset = 1
        # debugprint (f'  E x_offset {x_offset}')
    elif direction == ShipAction.WEST:
        x_offset = -1
        # debugprint (f'  W x_offset {x_offset}')
    # debugprint (f'  Next cell {direction} from {fromX} {fromY} is {fromX+x_offset} {fromY+y_offset}')
    return board[(fromX + x_offset, fromY + y_offset)]

def opposite(direction):
    if direction == ShipAction.NORTH:
        return ShipAction.SOUTH
    elif direction == ShipAction.SOUTH:
        return ShipAction.NORTH
    elif direction == ShipAction.EAST:
        return ShipAction.WEST
    elif direction == ShipAction.WEST:
        return ShipAction.EAST

# Check that ship in ship_cell is moving into target cell or not
def next_action_is_move_into_target(src_ship, ship_cell, target_cell):
    if ship_cell.ship is None: return False
    if ship_cell.ship.player_id is not board.current_player_id:
        # It is an enemy ship, check if it has higher halite than us.
        # If so, return False because we can clash and win.
        if src_ship is not None and ship_cell.ship.halite > src_ship.halite:
            return False
        else:
            srch = -1
            if src_ship:
                srch = src_ship.halite
            debugprint(f'    ship {ship_cell.ship.id} {ship_cell.ship.position} may move into {target_cell.position}'
                  f' shipH {ship_cell.ship.halite} srcH {srch}')
            return True
    if ship_states[ship_cell.ship.id] is ShipAction.CONVERT:
        # It is converting into shipyard so does not move
        debugprint(f'    ship {ship_cell.ship.id} {ship_cell.ship.position} convert, no move')
        return False
    if (
        ship_states[ship_cell.ship.id] == ShipAction.NORTH and
        ship_cell.north.position == target_cell.position
        ):
        debugprint(f'    ship {ship_cell.ship.id} {ship_cell.ship.position} move N to {target_cell.position}')
        return True
    elif (
        ship_states[ship_cell.ship.id] == ShipAction.SOUTH and
        ship_cell.south.position == target_cell.position
        ):
        debugprint(f'    ship {ship_cell.ship.id} {ship_cell.ship.position} move S to {target_cell.position}')
        return True
    elif (
        ship_states[ship_cell.ship.id] == ShipAction.EAST and
        ship_cell.east.position == target_cell.position
        ):
        debugprint(f'    ship {ship_cell.ship.id} {ship_cell.ship.position} move E to {target_cell.position}')
        return True
    elif (
        ship_states[ship_cell.ship.id] == ShipAction.WEST and
        ship_cell.west.position == target_cell.position
        ):
        debugprint(f'    ship {ship_cell.ship.id} {ship_cell.ship.position} move W to {target_cell.position}')
        return True
    return False

# Ship in the cell is moving away or not
def move_away(src_ship, cell):
    if cell.ship is None:
        debugprint(f'  cell {cell.position} ship null')
        return True
    # It is an enemy ship, check if it has higher halite than us.
    # If so, return True because we can clash and win.
    if cell.ship.player_id is not board.current_player_id:
        debugprint(f'    cell {cell.position} pid {cell.ship.player_id}'
              f' bpid {board.current_player_id} csH {cell.ship.halite}')
        if src_ship is not None and cell.ship.halite > src_ship.halite:
            return True
        else:
            return False
    # After this, we only deal with our ship.
    if (ship_states[cell.ship.id] in directions or
        ((src_ship is None or under_attack[src_ship.id]) and ship_states[cell.ship.id] is None)):
        # It moves away or hasn't been processed, we move in and force it to move.
        debugprint(f'    cell {cell.position} ship {cell.ship.id} move away {ship_states[cell.ship.id]}')
        return True
    else:
        debugprint(f'    cell {cell.position} ship {cell.ship.id} stay {ship_states[cell.ship.id]}')
        return False

def not_endgame_ok_to_clash():
    if obs.step < 394: return True
    elif obs.step >= 394: return False
    
def obstructed (ship, fromX, fromY, direction):
    nbr_cell = next_cell_in_direction(fromX[1], fromY[1], direction)
    if nbr_cell.shipyard is not None and our_shipyard(nbr_cell.shipyard):

        if (nbr_cell.ship and move_away(ship, nbr_cell) is False and not_endgame_ok_to_clash()
           ):
            # There is our ship already in the yard and it is not moving away so bail.
            return True

    # There are three nbrs of the nbr_cell. Loop through them and check that there is no ship there.
    for nbr_direction in directions:
        if nbr_direction is not opposite(direction):
            nbr_nbr_cell = next_cell_in_direction(nbr_cell.position.x, nbr_cell.position.y, nbr_direction)
            # debugprint (f'  Check nbr {nbr_cell.position} nbr2 {nbr_nbr_cell.position}')
            # Check if there is anything in the direction or moving into position we're heading
            if (
                (nbr_cell.ship and move_away(ship, nbr_cell) is False and 
                 (not_endgame_ok_to_clash() or our_ship(nbr_cell.ship) is False)) or
                (nbr_nbr_cell.ship and next_action_is_move_into_target(ship, nbr_nbr_cell, nbr_cell) and
                 (not_endgame_ok_to_clash() or our_ship(nbr_nbr_cell.ship) is False)) or
                (nbr_cell.shipyard and nbr_cell.shipyard.player_id is not board.current_player_id)        
               ):
                return True
    return False

stepAllowStrangeMove = 8

# Returns best direction to move from one position (fromPos) to another (toPos)
# Example: If I'm at pos 0 and want to get to pos 55, which direction should I choose?
# Do collision checking too. No point making any move only to crash
def getDirTo(ship, fromPos, toPos):
    global risk
    fromX, fromY = divmod(fromPos[0],size), divmod(fromPos[1],size)
    toX, toY = divmod(toPos[0],size), divmod(toPos[1],size)
    direction = None
    direction_ns = None
    direction_ew = None
    slant = False
    debugprint(f'   Want to go from {fromPos} to {toPos}')
    if fromY < toY:
        if abs(fromY[1]-toY[1]) <= size//2: direction_ns = ShipAction.NORTH
        else: direction_ns = ShipAction.SOUTH
    if fromY > toY:
        if abs(fromY[1]-toY[1]) <= size//2: direction_ns = ShipAction.SOUTH
        else: direction_ns = ShipAction.NORTH
    if fromX < toX:
        if abs(fromX[1]-toX[1]) <= size//2: direction_ew = ShipAction.EAST
        else: direction_ew = ShipAction.WEST
    if fromX > toX:
        if abs(fromX[1]-toX[1]) <= size//2: direction_ew = ShipAction.WEST
        else: direction_ew = ShipAction.EAST
    debugprint(f'   shortest to go {direction_ns} {direction_ew}')
    rd = risk[ship.id]
    
    if direction_ns is not None and direction_ew is not None:
        # if movement is possible on both x and y axis, choose the one with less risk first
        slant = True
        if rd[direction_ns] < rd[direction_ew]:
            direction = direction_ns
        elif rd[direction_ns] > rd[direction_ew]:
            direction = direction_ew
        elif random.randint(0,2) == 1:
            direction = direction_ew
        else:
            direction = direction_ns
    else:
        if direction_ns is not None:
            direction = direction_ns
        elif direction_ew is not None:
            direction = direction_ew

    if direction is None:
        # You are there already
        return direction
    
    # Check if there is anything in the direction we're heading
    if obstructed(ship, fromX, fromY, direction) is False:
        debugprint(f'   {direction} is unobstructed')
        return direction
    else:
        # try going around
        if direction == ShipAction.NORTH or direction == ShipAction.SOUTH:
            if direction_ew is not None:
                # The slant case
                new_direction = direction_ew
            else:
                # Choose east or west, the one with lower risk
                if rd[ShipAction.EAST] < rd[ShipAction.WEST]:
                    new_direction = ShipAction.EAST
                elif rd[ShipAction.EAST] > rd[ShipAction.WEST]:
                    new_direction = ShipAction.WEST
                elif random.randint(0,2) == 1:
                    new_direction = ShipAction.WEST
                else:
                    new_direction = ShipAction.EAST
        else:
            if direction_ns is not None:
                new_direction = direction_ns
            else:
                # Choose north or south, the one with lower risk
                if rd[ShipAction.NORTH] < rd[ShipAction.SOUTH]:
                    new_direction = ShipAction.NORTH
                elif rd[ShipAction.NORTH] > rd[ShipAction.SOUTH] :
                    new_direction = ShipAction.SOUTH
                elif random.randint(0,2) == 1:
                    new_direction = ShipAction.SOUTH
                else:
                    new_direction = ShipAction.NORTH
        debugprint(f'   {direction} is blocked, go {new_direction}')

        # Check again
        if obstructed(ship, fromX, fromY, new_direction) is False:
            debugprint(f'   {new_direction} is clear!')
            return new_direction
        else:
            if slant is True and obs.step > stepAllowStrangeMove:
                # No point going away from the target.
                # Allow in the initial stage to avoid blocking newly spawn ship.
                return None

            # try going around in the other direction
            debugprint(f'   {new_direction} is blocked, go opposite')
            new_direction = opposite(new_direction)
            # Check one more time
            if obstructed(ship, fromX, fromY, new_direction) is False:
                debugprint(f'   {new_direction} is clear!')
                return new_direction
            else:
                # Every direction is blocked
                debugprint(f'   {new_direction} is blocked!')
                return None

def steps_multiplying_factor(steps):
    if steps == 0: return 1
    elif steps == 1: return 1.6 # 1 step away has to be 55% better
    elif steps == 2: return 2   # 2 steps away has to be 100% better
    elif steps == 3: return 3   # 3 steps away has to be 200% better
    else: return 4 # Farther than 3 steps away has to be 300% better

# Returns the position in a grid centered at fromPos with highest Halite amount
# fromPos is from SDK so 0,0 is bottom left corner
# but obs.halite[coord] coord is for 0,0 at top left corner
def bestHalitePosition(ship):
    global ship_target
    target = None
    # If there is a specific target for this ship, just go there
    if obs.step < 20:
        target = ship_target[ship.id]
        if target is None:
            if len(s_env["target"]) > 0:
                pos_hal = s_env["target"].pop(0)
                ship_target[ship.id] = pos_hal[0]
                return pos_hal[0]
        else:
            return target

    highest_halite = highestX = highestY = highest_steps = 0
    toPos = fromPos = ship.position
    for pos in pos_halite:
        pos = fromPos+pos
        posX = pos.x % size
        posY = pos.y % size
        # cell_halite = obs.halite[size * (size-1-posY) + posX]
        cell_halite = s_env["map"][posX][size-1-posY]["halite"]
        # Ignore if our ship is already there
        if (
            board[pos].ship and
            board[pos].ship.id is not ship.id and
            board[pos].ship.player_id == board.current_player_id
           ):
            # debugprint(f'    Ignore {pos} our other ship already there')
            continue

        # debugprint(f'    H[{posX} {posY}] {cell_halite} Highest {highest_halite} Hstep {highest_steps}')
        steps = find_steps_between_pos(fromPos, pos)
        distance_factor = steps_multiplying_factor(steps-highest_steps)
        if cell_halite > highest_halite*distance_factor: # skew to prefer nearer cell
            highest_halite = cell_halite
            highestX = posX
            highestY = posY
            highest_steps = steps
    
    toPos = (highestX, highestY)
    return toPos

# Find a good place to create shipyard. Must have
# good halite in positions around it.
def good_halite_field_around_pos(position):
    total_h = 0
    for pos in one_step_away:
        pos = position+pos
        posX = pos.x % size
        posY = pos.y % size
        cell_halite = obs.halite[size * (size-1-posY) + posX]
        # Ignore if a ship is already there
        if board[pos].ship is None:
            total_h += cell_halite
    factor = 2 # if (obs.step < 280) else 2.5
    if total_h < s_env["average_halite"]*factor:
        return False
    return True

# Choose max halite nbr among all nbrs. May have zero.
def find_max_halite_nbr(position):
    max_nbr_ship = None
    max_dir = None
    for dir in directions:
        npos = position+one_step[dir]
        nship = board[npos].ship
        if (
            nship is not None and
            nship.player_id is board.current_player_id and
            (max_nbr_ship is None or nship.halite > max_nbr_ship.halite)
           ):
            max_nbr_ship = nship
            max_dir = dir
    return max_nbr_ship, max_dir

def nbr_ship_to_protect_base(ship_position):
    # Choose a nbr that has max halite
    max_nbr_ship, direction = find_max_halite_nbr(ship_position)
    if max_nbr_ship is not None:
        return max_nbr_ship, direction
    return None, None

# Return to base.
def return_to_base(ship, attacked):
    global next_pos

    target_shipyard, steps = find_nearest_shipyard(ship.position)
    if len(board.current_player.shipyards) == 0 or target_shipyard is None:
        # No shipyard to go to, the conversion has been or will be happening.
        # For this round, just stay put.
        if attacked:
            move_to_least_risky_direction(ship)
        return

    direction = getDirTo(ship, ship.position, target_shipyard.cell.position)
    if direction is None:
        # Path to the best shipyard is obstructed. Choose any other one.
        for shipyard in board.current_player.shipyards:
            if shipyard == target_shipyard:
                continue
            direction = getDirTo(ship, ship.position, shipyard.cell.position)
            if direction:
                break
    # We might still not have any good move to shipyard. That's ok just stay put.
    pos = ship.position
    if direction:
        ship_states[ship.id] = direction
        pos += one_step[direction]
        next_pos.append(round_position(pos))
    else:
        # If we decide to stay put, check that no one moves into our position.
        # If so, move based on risk-dir calculation
        if ship.position in next_pos or attacked:
            debugprint(f'  Ship {ship.id} {ship.player_id} {ship.position} has to move, A {attacked}')
            move_to_least_risky_direction(ship)
        else:
            next_pos.append(ship.position)
            ship_states[ship.id] = 'Processed'

def protect_shipyard():
    global next_pos
    next_pos = []
    for sy in board.current_player.shipyards:
        enemy_ship_steps = enemy_steps_from_shipyard(sy)
        if enemy_ship_steps == 1:
            # We should already have a ship on this shipyard. Make it stay put.
            ship = sy.cell.ship
            if ship:
                ship_states[ship.id] = 'Attacked'
                next_pos.append(ship.position)
                debugprint(f'  Ship {ship.id} stay put {ship.position} to block an attack')
            else:
                debugprint(f'  No ship on yard {sy.position} might be destroyed')
        elif enemy_ship_steps == 2:
            # If there is a ship wanting to deposit then
            #   move in. If there is a ship on yard then step out
            # elif find a our ship one step away
            #   move home
            # else spawn.
            ship = sy.cell.ship
            nbr_ship, direction = nbr_ship_to_protect_base(sy.position)
            if direction:
                # There is a nbr who wants to deposit
                for nship in board.current_player.ships:
                    if nship.id == nbr_ship.id:
                        nship.next_action = opposite(direction)
                ship_states[nbr_ship.id] = opposite(direction)
                next_pos.append(sy.position) # nbr steps into shipyard position
                debugprint(f'  Ship {nbr_ship.id} go to yard {sy.position} state {ship_states[nbr_ship.id]}')
            else:
                if ship:
                    # Stay put to protect the base
                    ship_states[ship.id] = 'Attacked'
                    next_pos.append(sy.position) # stay on the shipyard position
                    debugprint(f'  Ship {ship.id} on shipyard just stay put {ship.position}')
                else:
                    # Spawn to protect the yard if it's worth it
                    if spawn_to_protect_yard() and avg_getable_halite(sy.position) >= min_avg_halite_to_spawn():
                        shipyard_states[sy.id] = ShipyardAction.SPAWN
                        debugprint(f'  Shipyard {sy.id} {sy.position} spawn to protect yard')

def spawn_to_protect_yard():
    num_shipyard = len(board.current_player.shipyards)
    if obs.step < 275:
        # Protect the last three
        if num_shipyard < 4:
            return True
    elif obs.step < 320:
        # Protect the last two
        if num_shipyard < 3:
            return True
    elif obs.step < 360:
        # Protect the last one
        if num_shipyard < 2:
            return True
    return False

def game_steps_left():
    return 399-obs.step

# Returns the position in a grid centered at fromPos with highest Halite amount
def bestJuicyEnemy(ship):
    highest_halite = highest_steps = num_enemy_ships = 0
    fromPos = ship.position
    highestX = fromPos.x % size
    highestY = fromPos.y % size
    for pos in pos_enemy:
        pos = fromPos+pos
        posX = pos.x % size
        posY = pos.y % size
        # Look at enemy ships that have halite
        if (
            board[pos].ship and
            board[pos].ship.player_id != board.current_player_id
           ):
            num_enemy_ships += 1
            if board[pos].ship.halite > 0 and board[pos].ship.halite < 500:
                # Ignore ship with higher than 500 because it will convert to base
                debugprint(f'    H[{posX} {posY}] {board[pos].ship.halite} Highest {highest_halite} Hstep {highest_steps}')
                steps = find_steps_between_pos(fromPos, pos)
                distance_factor = steps_multiplying_factor(steps-highest_steps)
                if board[pos].ship.halite > highest_halite*distance_factor: # skew to prefer nearer cell
                    highest_halite = board[pos].ship.halite
                    highestX = posX
                    highestY = posY
                    highest_steps = steps
    toPos = (highestX, highestY)
    return toPos, num_enemy_ships, highest_halite

def total_cargo():
    cargo = 0
    for ship in board.current_player.ships:
        cargo += ship.halite
    return cargo

# Return True if there is an enemy ship with lower halite than this ship's within 2 steps
def enemy_nearby(ship):
    for pos in two_steps_away:
        pos = ship.position+pos
        posX = pos.x % size
        posY = pos.y % size
        # Look for enemy ships that have halite less than this ship's
        if (
            board[pos].ship and
            board[pos].ship.player_id != board.current_player_id and
            board[pos].ship.halite < ship.halite+ship.cell.halite/4
           ):
            return True
    return False

def position_central(position):
    me = board.current_player
    posX = position.x % size
    posY = position.y % size
    if len(me.shipyards) < max_shipyards()-1:
        # If we have fewer yard than anticipated then allow non-central yard
        return True
    if posX >= 3 and posX <= 17 and posY >= 3 and posY <= 17:
        return True
    return False

def create_shipyard_when_zero():
    if obs.step < 300:
        return True
    elif obs.step < 350:
        if total_cargo() > 500:
            return True
    elif obs.step < 370:
        if total_cargo() > 600 and s_env["average_halite"] >= min_avg_halite_to_convert():
            return True
    elif s_env["average_halite"] >= min_avg_halite_to_convert():
        return True
    return False

def process_non_attacked_ship_sub(ship):
    global new_yard_create
    global next_pos
    global under_attack
    global curr_hunting_mode

    me = board.current_player

    debugprint(f'  Ship {ship.id} {ship.player_id} {ship.position} '
          f'St {ship_states[ship.id]} H {ship.halite} cH {ship.cell.halite}')
    myHalite = board.players[board.current_player_id].halite

    # If there is no base, let's create one ASAP
    if (
        len(me.shipyards) == 0 and
        ship.halite+myHalite >= 500 and
        new_yard_create == False and
        create_shipyard_when_zero() and
        position_central(ship.position)
       ):
        debugprint(f'  No shipyard: ship {ship.position} H{ship.halite} MH{myHalite} convert')
        ship.next_action = ShipAction.CONVERT
        ship_states[ship.id] = ShipAction.CONVERT
        new_yard_create = True
        return

    # If cargo gets big AND where we are is low, deposit at base
    minHaliteTakeHome = min_halite_take_home()
    minCellHaliteToDig = min_cell_halite_to_dig()
    haliteHoldLimit = halite_hold_limit()
    target_shipyard, step_to_nearest_shipyard = find_nearest_shipyard(ship.position)
    if (target_shipyard and
        ((game_steps_left() <= step_to_nearest_shipyard+1) or
         (ship.halite > minHaliteTakeHome and
          (ship.cell.halite < minCellHaliteToDig or ship.halite > haliteHoldLimit)) or
         (curr_hunting_mode and ship.halite > 0 and enemy_nearby(ship)))
        # This can cause oscillation when we carry lots
        # but the current cell is high in halite so we
        # don't return to base and fall through. This
        # means we step toward best halite cell but the
        # intermediate step is low so we try to go home.
       ):
        debugprint(f'  minHTH {minHaliteTakeHome} minCHTD {minCellHaliteToDig} HHL {haliteHoldLimit} go to base')
        # If the nearest shipyard is too far and there are enough to convert AND spawn, convert this ship
        maxStepHome = max_step_home()
        maxShipyards = max_shipyards()
        minAvgHaliteToConvert = min_avg_halite_to_convert()
        if (
            ship.halite > 100 and
            ship.halite+myHalite > min_halite_to_convert() and new_yard_create is False and
            target_shipyard is not None and
            step_to_nearest_shipyard >= maxStepHome and
            step_to_nearest_shipyard < 12 and
            len(me.shipyards) < maxShipyards and
            s_env["average_halite"] >= minAvgHaliteToConvert and
            good_halite_field_around_pos(ship.position) and
            position_central(ship.position)
            ):
            debugprint(f'  ship {ship.position} H{ship.halite} MH{myHalite}'
                  f' convert maxstep {maxStepHome} maxship {maxShipyards}')
            ship.next_action = ShipAction.CONVERT
            ship_states[ship.id] = ShipAction.CONVERT
            new_yard_create = True
        else:
            return_to_base(ship, False) # Handle the case when there is no base or path is obstructed
        return

    # We are collecting. Move to the best location.
    stay_put_to_dig = False
    if curr_hunting_mode and ship.halite == 0:
        best_position, num_enemy_ships, highest_halite = bestJuicyEnemy(ship)
        debugprint(f'Step {obs.step} ship {ship.id} {ship.position} hunting.'
                   f' juicy {best_position} numEn {num_enemy_ships} highH {highest_halite} cellH {ship.cell.halite}')
        if num_enemy_ships == 0:
            # There is no enemy around, we should really take a quick dig.
            if (ship.cell.halite >= 250):
                stay_put_to_dig = True
                best_position = ship.position
            else:
                best_position = bestHalitePosition(ship)
        elif highest_halite == 0:
            # No enemy ship in sight, let's go near best halite cell
            best_position = bestHalitePosition(ship)
    else:
        best_position = bestHalitePosition(ship)
    direction = getDirTo(ship, ship.position, best_position)
    pos = ship.position
    if direction:
        ship_states[ship.id] = direction
        pos += one_step[direction]
        next_pos.append(round_position(pos))
    else:
        # If we decide to stay put, check that no one moves into our position.
        # If so, move based on risk-dir calculation
        if (
            ship.position in next_pos or
            (curr_hunting_mode and ship.halite == 0 and stay_put_to_dig == False)
           ):
            debugprint(f'  Ship {ship.id} {ship.player_id} {ship.position} H {ship.halite} has to move')
            move_to_least_risky_direction(ship)
        else:
            next_pos.append(ship.position)
            ship_states[ship.id] = 'Processed'

    # Set the target cell to 0 so other ships won't go there
    posX = best_position[0] % size
    posY = best_position[1] % size
    s_env["map"][posX][size-1-posY]["halite"] = 0

def process_non_attacked_ships():
    global new_yard_create
    global next_pos
    global under_attack
    me = board.current_player
    debugprint(f'  NextPosList b4 non-attacked {next_pos}')
    # Process all ships in low-to-high order of risk
    sorted_rs = sorted(risksum.items(), key=lambda item: item[1])

    # Process ship on next_pos in the first round
    for id_risk in sorted_rs:
        shipid = id_risk[0]
        if ship_states[shipid] is not None:
            # This ship has already been processed, skip it
            continue
        if under_attack[shipid]:
            # Already process attacked ships
            continue
        ship = ship_by_id(shipid)
        assert(ship)
        if ship.position in next_pos:
            process_non_attacked_ship_sub(ship)

    # Process other non-attacked ships in the second round
    for id_risk in sorted_rs:
        shipid = id_risk[0]
        if ship_states[shipid] is not None:
            # This ship has already been processed, skip it
            continue
        if under_attack[shipid]:
            # Already process attacked ships
            continue
        ship = ship_by_id(shipid)
        assert(ship)
        process_non_attacked_ship_sub(ship)

def round_position(position):
    posX = position[0] % size
    posY = position[1] % size
    return ((posX,posY))

def move_to_least_risky_direction(ship):
    global next_pos
    global under_attack

    # min_risk_dir = min(risk[ship.id].values())
    # if min_risk_dir >= 100:
    if under_attack[ship.id]:
        # Convert if we might be attacked and carry more than 500 halite.
        # Net gain is minimal but better than let the enemy have it.
        if ship.halite >= 500:
            debugprint(f'  Convert H {ship.halite}')
            ship.next_action = ShipAction.CONVERT
            ship_states[ship.id] = ShipAction.CONVERT
            return

    # Move to the least risky direction based on risk
    sorted_risk_dir = sorted(risk[ship.id].items(), key=lambda item: item[1])
    debugprint(f'  sr {sorted_risk_dir}')
    for dir_risk in sorted_risk_dir:
        dir = dir_risk[0]
        npos = round_position(ship.position+one_step[dir])
        debugprint(f' ship {ship.id} want to go {dir} to {npos}')
        debugprint(f'   Next_pos {next_pos}')
        # Check that no other ship is moving into the position
        if npos not in next_pos:
            ship_states[ship.id] = dir
            next_pos.append(npos)
            debugprint(f' ship {ship.id} go {dir} to {npos}')
            break
        # If all direction is blocked then we are out of luck

def process_attacked_ships():
    global under_attack
    # Process all ships in high-to-low order of risk so higher risk ships move first
    debugprint(f'  NextPosList b4 attacked {next_pos}')
    sorted_rs = sorted(risksum.items(), key=lambda item: item[1], reverse=True)
    for id_risk in sorted_rs:
        shipid = id_risk[0]
        if ship_states[shipid] is not None:
            # This ship has already been processed, skip it
            continue
        if under_attack[shipid] is False:
            # skip non-attacked ships
            continue
        ship = ship_by_id(shipid)
        assert(ship)
        debugprint(f'  Ship {ship.id} {ship.player_id} {ship.position} '
              f'St {ship_states[ship.id]} H {ship.halite} cH {ship.cell.halite}')
        return_to_base(ship, True)

def avg_cell_halite_in_free_square(pos,sqsize):
    posX = pos.x % size
    posY = pos.y % size
    totalH = numCell = 0
    offset = sqsize//2
    debugprint(f'  Shipyard pos {pos} {sqsize}')
    for x in range(0,sqsize):
        for y in range(0,sqsize):
            xx = posX+x-offset
            yy = posY+y-offset
            if xx < 0: xx += size
            if yy < 0: yy += size
            xx = xx % size
            yy = yy % size
            curpos = (xx,yy)
            debugprint(f' Look for halite at pos {curpos}')
            if board[curpos].ship is None:
                cell_halite = obs.halite[size * (size-1-yy) + xx]
                totalH += cell_halite
                numCell += 1
                debugprint(f' Halite at pos {numCell} {curpos} {cell_halite} {totalH}')
    return totalH/numCell

def avg_getable_halite(position):
    if obs.step < 250:
        return s_env["average_halite"]
    elif obs.step < 350:
        return avg_cell_halite_in_free_square(position,12)
    else:
        return avg_cell_halite_in_free_square(position,8)

def process_shipyards():
    me = board.current_player
    spawn_count = 0
    for sy in reversed(me.shipyards):
        if sy.next_action is not None:
            # We might already spawn to protect this yard
            continue
        # If there is our ship moving into this shipyard, do not spawn
        if (
            next_action_is_move_into_target(None, sy.cell.north, sy.cell) or
            next_action_is_move_into_target(None, sy.cell.south, sy.cell) or
            next_action_is_move_into_target(None, sy.cell.east, sy.cell) or
            next_action_is_move_into_target(None, sy.cell.west, sy.cell)
           ):
            debugprint(f'  Something moves into this yard {sy.position}')
            continue
        # If the ship on the yard does not move out because we might be attacked, do not spawn
        yship = sy.cell.ship
        if (
            move_away(None, sy.cell) is False and yship is not None
           ):
            debugprint(f'  ship {yship.id} not moving away from this yard {sy.position}')
            continue
            
        # As we approach end game, we lower max ships.
        if (
            len(me.ships) < max_ship() and 
            me.halite-(spawn_count*500) >= min_halite_to_spawn() and
            (avg_getable_halite(sy.position) >= min_avg_halite_to_spawn())
           ):
            shipyard_states[sy.id] = ShipyardAction.SPAWN
            spawn_count += 1
            debugprint(f'  Ship {len(me.ships)} Max {max_ship()} Spawn {spawn_count}')

# Goes through list of ship sorted by risksum
def ships_actions():
    global new_yard_create
    global ship_target
    new_yard_create = False
    debugprint(f'Step {obs.step} pid {board.current_player_id} '
          f'Shipyard {len(board.current_player.shipyards)} Ship {len(board.current_player.ships)}')
    debugprint(f'  curH {board.players[board.current_player_id].halite} avgH {s_env["average_halite"]}')
    debugprint(board)
    # Make sure that ship_states array exists for all ships
    for ship in board.current_player.ships:
        ship_states[ship.id] = None
    for shipyard in board.current_player.shipyards:
        shipyard_states[shipyard.id] = None
        sy_ship = shipyard.cell.ship
        if sy_ship and sy_ship.halite == 0:
            # Clear target for ship that has finished deposit
            ship_target[sy_ship.id] = None

    # Zero pass deals with protecting shipyard.
    protect_shipyard()
    process_attacked_ships()
    process_non_attacked_ships()
    process_shipyards()

def our_num_ship_top_two():
    global num_ship
    our_num_ship = sum(num_ship[board.current_player.id])/len(num_ship[board.current_player.id])
    oppo_better = 0
    for player in board.opponents:
        oppo_num_ship = sum(num_ship[player.id])/len(num_ship[player.id])
        if our_num_ship < oppo_num_ship:
            oppo_better += 1
            if oppo_better == 2:
                return False
    return True

curr_hunting_mode = False
def hunting_mode():
    global aggressor
    global total_agg_ships
    # if obs.step > 73 and obs.step < 320: return True
    # else: return False
    if obs.step >= 320: return False
    elif obs.step >= 49:
        debugprint(f'Step {obs.step} {aggressor} aggs, Top2 {our_num_ship_top_two()}, '
              f'running avg hal {s_env["running_average_halite"]} LINE {min_avg_halite_to_spawn()}')
        if aggressor == 0:
            return False
        elif aggressor == 1:
            offset = 0
            if total_agg_ships <= 10:
                offset = 0
            elif total_agg_ships <= 15:
                offset = 10
            elif total_agg_ships <= 20:
                offset = 20
            else:
                return True
            if (
                s_env["running_average_halite"] > min_avg_halite_to_spawn()+offset or
                s_env["running_average_halite"] < min_avg_halite_to_spawn()-offset-5
               ):
                # If avg halite is high or very low, let's dig
                return False
            else:
                # Continue hunting
                return True
        else:
            # 2-3 aggressors, just hunt
            return True
    else: return False
    
def min_halite_to_spawn():
    if obs.step < 60: return 500
    elif obs.step < 80: return 550
    elif obs.step < 100: return 600
    elif obs.step < 125: return 700
    elif obs.step < 150: return 800
    elif obs.step < 175: return 900
    else: return 1000

def min_avg_halite_to_spawn():
    return 32+(obs.step/5.88) # 32-100

def min_halite_to_convert():
    if obs.step < 30: return 500
    elif obs.step < 73: return 600
    else: return 700

def min_avg_halite_to_convert():
    return min_avg_halite_to_spawn()

def halite_hold_limit():
    if obs.step < 25: return 500
    elif obs.step < 65: return 300
    elif obs.step < 73: return 64 if curr_hunting_mode else 250
    elif obs.step < 100: return 64 if curr_hunting_mode else 100
    elif obs.step < 200: return 64 if curr_hunting_mode else 120
    elif obs.step < 280: return 64 if curr_hunting_mode else 200
    elif obs.step < 320: return 64 if curr_hunting_mode else 300
    elif obs.step < 350: return 350
    elif obs.step < 385: return 275
    else: return 100

def min_cell_halite_to_dig():
    if obs.step < 100: return 100
    elif obs.step < 200: return 70
    elif obs.step < 320: return 30
    elif obs.step < 350: return 20
    elif obs.step < 385: return 10
    elif obs.step < 395: return 50
    else: return 100 # In end game, we want to deposit more

def min_halite_take_home():
    if obs.step < 25: return 450
    elif obs.step < 50: return 250
    elif obs.step < 73: return 64 if curr_hunting_mode else 200
    elif obs.step < 100: return 64 if curr_hunting_mode else 100
    elif obs.step < 200: return 64 if curr_hunting_mode else 120
    elif obs.step < 280: return 64 if curr_hunting_mode else 200
    elif obs.step < 320: return 64 if curr_hunting_mode else 250
    elif obs.step < 350: return 300
    elif obs.step < 385: return 250
    else: return 100

def max_step_home():
    if obs.step < 75: return 6
    elif obs.step < 320: return 6
    elif obs.step < 350: return 8
    else: return 10

def max_ship():
    if obs.step < 50: return 25
    elif obs.step < 150: return 35 if curr_hunting_mode else 30
    elif obs.step < 200: return 40 if curr_hunting_mode else 35
    elif obs.step < 250: return 45 if curr_hunting_mode else 40
    elif obs.step < 320: return 40 if curr_hunting_mode else 35
    elif obs.step < 350: return 35 if curr_hunting_mode else 30
    else: return 1

def max_shipyards():
    if obs.step < 50: return 2
    elif obs.step < 73: return 3
    elif obs.step < 100: return 3 if curr_hunting_mode else 2
    elif obs.step < 350: return 4 if curr_hunting_mode else 2
    else: return 1

# This initially came from https://www.kaggle.com/yegorbiryukov/halite-swarm-intelligence
def get_map_and_average_halite():
    game_map = []
    halite_sources_amount = 0
    halite_total_amount = 0
    for x in range(size):
        game_map.append([])
        for y in range(size):
            game_map[x].append({
                # amount of halite
                "halite": obs.halite[size * y + x]
            })
            # Ignore zero halite cell
            if obs.halite[size * y + x] > 0:
                halite_total_amount += obs.halite[size * y + x]
                halite_sources_amount += 1
    average_halite = halite_total_amount / halite_sources_amount
    return game_map, average_halite

# In the beginning phase, choose the best 9 halite positions to
# assign to the first nine ships.
def find_target_halite_position():
    hal_list = {}
    for sy in board.current_player.shipyards:
        pos = sy.position
        debugprint(f'  Shipyard pos {pos}')
        posX = pos.x % size
        posY = pos.y % size
        for x in range(posX-5,posX+6):
            for y in range(posY-5,posY+6):
                cell_halite = s_env["map"][x][size-1-y]["halite"]
                debugprint(f'Cell ({x},{y}) H: {cell_halite}')
                if cell_halite < 300 and x in range(posX-1,posX+2) and y in range(posY-1,posY+2):
                    # Skip those near the base to mine later
                    continue
                hal_list[(x,y)] = cell_halite
        top9 = sorted(hal_list.items(), key=lambda item: item[1], reverse=True)
        top9 = top9[:9]
        debugprint(f' Top 9 {top9}')
        return top9

aggressor = 0
total_agg_ships = 0
def setup():
    global aggressor
    global total_agg_ships
    global zero_hal
    global running_avg_halite
    global num_ship
    global curr_hunting_mode

    s_env["map"], s_env["average_halite"] = get_map_and_average_halite()
    if obs.step == 1:
        # Create best 9 halite target positions for the first 9 ships
        s_env["target"] = find_target_halite_position()

    if len(running_avg_halite) == 10:
        running_avg_halite.pop(0)
    running_avg_halite.append(s_env["average_halite"])
    s_env["running_average_halite"] = sum(running_avg_halite)/len(running_avg_halite)
    debugprint(f'Player {board.current_player.id} running_avg_halite {running_avg_halite} = {s_env["running_average_halite"]}')

    for key, player in board.players.items():
        if len(num_ship[player.id]) == 10:
            num_ship[player.id].pop(0)
        num_ship[player.id].append(len(player.ships))
    
    # Calculate last 5 steps running avg. of zero halite ships of the three enemy
    if obs.step >= 40 and obs.step <= 320:
        for player in board.opponents:
            count = 0
            if len(zero_hal[player.id]) == 5:
                zero_hal[player.id].pop(0)
            for ship in player.ships:
                if ship.halite == 0:
                    count += 1
            zero_hal[player.id].append(count)
            debugprint(f'  Step {obs.step}: Player {player.id} zeroH count {count} {zero_hal[player.id]}')
        if obs.step >= 49 and obs.step <= 320:
            aggressor = total_agg_ships = 0
            # Calculate avg. of each player and declare if it is an aggressor or not
            for player in board.opponents:
                avg = sum(zero_hal[player.id])/len(zero_hal[player.id])
                debugprint(f'  Step {obs.step}: Player {player.id} {zero_hal[player.id]} avg {avg}')
                if avg > 9:
                    aggressor += 1
                    total_agg_ships += avg
            debugprint(f'Num Agg {aggressor} ships {total_agg_ships}')
    curr_hunting_mode = hunting_mode()

def debugprint_none(*args):
  pass

def init(obs,config):
    # This is only called on first call to agent()
    # Do initalization things
    global debugprint
    global zero_hal
    global num_ship
    global running_avg_halite
    if not quiet:
        #we are called locally, so leave debugprints OK
        #pass
        debugprint=print
        pprint.pprint=print
    else:
        #we are called in competition, quiet output
        debugprint=debugprint_none
        pprint.pprint=debugprint_none
    zero_hal.clear()
    num_ship.clear()
    running_avg_halite.clear()
    for x in range(0,4):
        zero_hal.append([])
        num_ship.append([])

zero_hal = []
num_ship = []
running_avg_halite = []
board = None
obs = None
config = None
size = None

def calculate_risk_all_ships():
    global risk
    global risksum
    global under_attack

    risk = {}
    risksum = {}
    under_attack = {}
    for ship in board.current_player.ships:
        rdlist = {}
        sum = 0
        under_attack[ship.id] = False
        for dir in directions:
            rk = calculate_risk(ship, dir)
            rdlist[dir] = rk
            sum += rk
        risk[ship.id] = rdlist
        risksum[ship.id] = sum

# Enemy ship has higher risk value
one_step_risk_values = {True: 20, False: 100}
one_step_shipyard_risk_values = {True: 0, False: 130}
two_step_risk_values = {True: 10, False: 101}
two_step_shipyard_risk_values = {True: 0, False: 20}

def calculate_risk(ship, direction):
    global under_attack

    risk_val = 0
    npos = ship.position+one_step[direction]
    nship = board[npos].ship
    if nship:
        if (
            our_ship(nship) or
            (hunting_mode() is True and nship.halite <= ship.halite) or
            (hunting_mode() is False and nship.halite < ship.halite)
           ):
            risk_val += one_step_risk_values[our_ship(nship)]
            if our_ship(nship) is False:
                under_attack[ship.id] = True
            debugprint(f'UA {under_attack[ship.id]} ship {ship.id} {ship.position} h {ship.halite} nbr {nship.id} {nship.position} h {nship.halite}')
        else:
            # Fudge to handle the case when we beat one-step enemy but
            # lose to two-step ones.
            risk_val += 1
        debugprint(f'  {ship.id} 1-step Risk {direction} {risk_val}')
    nsy = board[npos].shipyard
    if nsy:
        risk_val += one_step_shipyard_risk_values[our_shipyard(nsy)]
        debugprint(f'  {ship.id} 1-step SY Risk {direction} {risk_val}')

    two_steps_list = two_steps[direction]
    for pos in two_steps_list:
        npos = ship.position+pos
        nship = board[npos].ship
        if nship:
            if (
                our_ship(nship) or
                (hunting_mode() is True and nship.halite <= ship.halite) or
                (hunting_mode() is False and nship.halite < ship.halite)
               ):
                risk_val += two_step_risk_values[our_ship(nship)]
            else:
                # Fudge so we choose the less crowded dir
                risk_val += 1
            debugprint(f'  {ship.id} 2-step Risk {direction} {risk_val}')
        nsy = board[npos].shipyard
        if nsy:
            risk_val += two_step_shipyard_risk_values[our_shipyard(nsy)]
            debugprint(f'  {ship.id} 2-step SY Risk {direction} {risk_val}')
    return risk_val

did_init=False
quiet=True
start=None

# Returns the commands we send to our ships and shipyards
def agent(observation, configuration):
    global did_init
    global start
    global board
    global obs
    global config
    global size

    obs = observation
    config = configuration
    size = config.size
    #Do initialization 1 time
    start_step=time.time()
    if start is None:
        start=time.time()
    if not did_init:
        init(obs,config)
        did_init=True

    board = Board(obs, config)
    me = board.current_player
    setup()
    calculate_risk_all_ships()
    debugprint(f'Under attack {under_attack}')
    ships_actions()

    ##############
    epsilon = conf.get('AGENT2_EPSILON')
    for ship in me.ships:
        debugprint(f'  ShipStates {ship_states[ship.id]}')
        if ship_states[ship.id] in directions:
            ship.next_action = ship_states[ship.id]
        if epsilon > np.random.rand() and board.step > 5:
            action = np.random.choice(5)
            ship.next_action = helper.ship_action_string(action)
    for shipyard in me.shipyards:
        debugprint(f'  ShipYardStates {shipyard_states[shipyard.id]}')
        shipyard.next_action = shipyard_states[shipyard.id]
    ##############

    debugprint(f' Step {obs.step} shipAct {me.next_actions}')

    debugprint('Time this turn: {:8.3f} total elapsed {:8.3f}'.format(time.time()-start_step,time.time()-start))
    return me.next_actions