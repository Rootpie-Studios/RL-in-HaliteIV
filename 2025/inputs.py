import numpy as np
import helper


def obs_to_transformer_input(
    obs,
    control_ship_id,
    player_id,
    own_ship_actions,
    size=21,
    episode_steps=400,
    debugger=None,
):
    """Creates a transformer-friendly input representation for a single ship.
    Returns:
        - state_matrices: Array of shape [1, size, size, channels] containing:
            - Channel 0: Halite amounts normalized
            - Channel 1: Friendly ships (1 for ship presence)
            - Channel 2: Friendly shipyards (1 for shipyard)
            - Channel 3: Enemy ships (1 for ship presence)
            - Channel 4: Enemy shipyards (1 for shipyard)
            - Channel 5: Ship cargo (normalized cargo amounts)
        - global_features: Array of shape [1, global_features] containing:
            - Current ship's cargo
            - Current step normalized
            - Player halite
            - Enemy halite
    """
    if debugger:
        debugger.write(f"\n=== Input Processing for Ship {control_ship_id} ===\n")
        debugger.write(f"Player ID: {player_id}\n")

    own_ships = obs["players"][player_id][2]
    control_ship_data = own_ships[control_ship_id]
    enemy_id = 1 if player_id == 0 else 0

    if debugger:
        debugger.write(
            f"Ship Position: {control_ship_data[0]}, Cargo: {control_ship_data[1]}\n"
        )
        debugger.write(f"Number of friendly ships: {len(own_ships)}\n")
        debugger.write(f"Number of enemy ships: {len(obs['players'][enemy_id][2])}\n")

    # Initialize state matrices
    halite_map = np.array([x / 500 for x in obs["halite"]])
    friendly_ships = np.zeros(size * size)
    friendly_yards = np.zeros(size * size)
    enemy_ships = np.zeros(size * size)
    enemy_yards = np.zeros(size * size)
    cargo_map = np.zeros(size * size)

    # Fill in ship and shipyard positions
    for idx, player in enumerate(obs["players"]):
        ships = player[2]
        shipyards = player[1]

        if debugger:
            debugger.write(f"\nProcessing Player {idx}:\n")
            debugger.write(f"Ships: {len(ships)}, Shipyards: {len(shipyards)}\n")

        for ship_id, ship_data in ships.items():
            if idx == player_id:
                if ship_id != control_ship_id:
                    friendly_ships[ship_data[0]] = 1
                cargo_map[ship_data[0]] = ship_data[1] / 1000
            else:
                enemy_ships[ship_data[0]] = 1
                cargo_map[ship_data[0]] = ship_data[1] / 1000

        for _, yard_pos in shipyards.items():
            if idx == player_id:
                friendly_yards[yard_pos] = 1
            else:
                enemy_yards[yard_pos] = 1

    # Center all matrices on the controlled ship
    ship_pos = control_ship_data[0]
    matrices = [
        helper.center_cell(halite_map, ship_pos),
        helper.center_cell(friendly_ships, ship_pos),
        helper.center_cell(friendly_yards, ship_pos),
        helper.center_cell(enemy_ships, ship_pos),
        helper.center_cell(enemy_yards, ship_pos),
        helper.center_cell(cargo_map, ship_pos),
    ]

    # Stack the matrices into channels and reshape into a 2D grid.
    state_matrices = np.stack(matrices, axis=-1)  # shape: (size*size, channels)
    state_matrices = state_matrices.reshape(
        size, size, -1
    )  # reshape to (size, size, channels)

    # Add batch and time dimensions (for now, time_steps=1)
    state_matrices = np.expand_dims(state_matrices, axis=0)  # batch dimension
    state_matrices = np.expand_dims(state_matrices, axis=1)  # time dimension

    # Global features: ensure shape (1, 1, num_global_features)
    global_features = np.array(
        [
            control_ship_data[1] / 1000,  # Current ship cargo
            obs["step"] / episode_steps,  # Normalized step
            obs["players"][player_id][0] / 5000,  # Player halite
            obs["players"][enemy_id][0] / 5000,  # Enemy halite
        ]
    )
    global_features = np.expand_dims(global_features, axis=0)  # batch dimension
    global_features = np.expand_dims(global_features, axis=1)  # time dimension

    # Debug logging remains unchanged.
    if debugger:
        debugger.write("\nGlobal Features:\n")
        debugger.write(f"Ship Cargo: {control_ship_data[1]/1000:.3f}\n")
        debugger.write(f"Current Step: {obs['step']/episode_steps:.3f}\n")
        debugger.write(f"Player Halite: {obs['players'][player_id][0]/5000:.3f}\n")
        debugger.write(f"Enemy Halite: {obs['players'][enemy_id][0]/5000:.3f}\n")
        debugger.write("=== End Input Processing ===\n")

    return state_matrices, global_features
