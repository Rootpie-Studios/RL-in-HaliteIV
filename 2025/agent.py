import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from inputs import obs_to_transformer_input
from helper import ship_action_string, get_next_ship_pos
from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction
from copy import deepcopy
import random
import conf


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class HaliteAgent(keras.Model):
    def __init__(
        self,
        board_size=21,
        num_channels=6,
        num_global_features=4,
        embed_dim=64,
        num_transformer_blocks=6,
        num_heads=8,
        ff_dim=64,
        dropout_rate=0.1,
        max_time_steps=10,
    ):
        super(HaliteAgent, self).__init__()

        # Define input layers
        self.state_input = keras.Input(
            shape=(None, board_size, board_size, num_channels), name="state_matrices"
        )
        self.global_input = keras.Input(
            shape=(None, num_global_features), name="global_features"
        )

        # Use TimeDistributed wrappers for frame-wise processing:
        self.td_conv1 = layers.TimeDistributed(
            layers.Conv2D(32, 3, padding="same", activation="relu")
        )
        self.td_conv2 = layers.TimeDistributed(
            layers.Conv2D(64, 3, padding="same", activation="relu")
        )
        self.td_pooling = layers.TimeDistributed(layers.GlobalAveragePooling2D())

        # Process global features per frame
        self.td_global_dense = layers.TimeDistributed(
            layers.Dense(embed_dim, activation="relu")
        )

        # Project combined frame features to embed_dim
        self.frame_projection = layers.Dense(embed_dim, activation="relu")

        # Time positional embedding (for the time axis)
        self.time_position_embedding = layers.Embedding(
            input_dim=max_time_steps, output_dim=embed_dim
        )

        # Transformer blocks remain unchanged.
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_transformer_blocks)
        ]

        # Output layers – note we take the last time step's representation.
        self.final_dense1 = layers.Dense(128, activation="relu")
        self.final_dense2 = layers.Dense(6)  # 5 moves + convert

    def call(self, inputs):
        state_matrices = inputs["state_matrices"]
        global_features = inputs["global_features"]

        batch_size = tf.shape(state_matrices)[0]
        time_steps = tf.shape(state_matrices)[1]

        # Process each frame spatially using TimeDistributed convs
        x = self.td_conv1(state_matrices)
        x = self.td_conv2(x)
        spatial_features = self.td_pooling(x)

        # Process global features per frame
        global_features = self.td_global_dense(
            global_features
        )  # (batch, time_steps, embed_dim)

        # Combine spatial and global features per frame
        combined_features = tf.concat([spatial_features, global_features], axis=-1)
        combined_features = self.frame_projection(
            combined_features
        )  # (batch, time_steps, embed_dim)

        # Add time positional encoding
        time_positions = tf.range(start=0, limit=time_steps, delta=1)
        time_positions = self.time_position_embedding(
            time_positions
        )  # (time_steps, embed_dim)
        time_positions = tf.expand_dims(
            time_positions, axis=0
        )  # (1, time_steps, embed_dim)
        combined_features += time_positions

        # Apply transformer blocks over the sequence of frames
        for transformer_block in self.transformer_blocks:
            combined_features = transformer_block(combined_features, training=True)

        # Use the last time-step representation for Q-value prediction
        x = combined_features[:, -1, :]
        x = self.final_dense1(x)
        x = self.final_dense2(x)

        return x


def create_agent():
    """Creates and compiles the Halite agent model"""
    model = HaliteAgent()

    # Build the model with sample inputs
    sample_state = np.zeros((1, 1, 21, 21, 6))
    sample_global = np.zeros((1, 1, 4))
    model({"state_matrices": sample_state, "global_features": sample_global})

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["accuracy"],
    )
    return model


def explore_branch(
    obs, config, ship_id, action, depth=10, num_branches=3, debugger=None
):
    """Explores a possible future branch by simulating suboptimal actions"""
    board = Board(obs, config)

    if debugger:
        debugger.write(f"Exploring branch {action} at depth {depth}\n")

    # Initialize metrics for this branch
    total_reward = 0
    discount = 0.95

    # Run simulation for specified depth
    for step in range(depth):
        # Take the specified action for the first step
        actions = {}
        if step == 0:
            actions[ship_id] = ship_action_string(action)
            if debugger:
                debugger.write(
                    f"Taking action {ship_action_string(action)} at step {step}\n"
                )
        try:
            for ship_id, ship in board.ships.items():
                if ship.id not in actions:
                    if debugger:
                        debugger.write(f"Processing ship {ship.id}\n")
                    state_matrices, global_features = obs_to_transformer_input(
                        obs,
                        ship.id,
                        ship.player.id,
                        actions,
                        debugger=debugger,
                    )

                    inputs = {
                        "state_matrices": state_matrices,
                        "global_features": global_features,
                    }
                    q_values = agent.model.predict(inputs)
                    if debugger:
                        debugger.write(f"Q-values: {q_values[0]}\n")
                    top_k_actions = np.argsort(q_values[0])[-num_branches:]
                    chosen_action = np.random.choice(top_k_actions)
                    actions[ship.id] = ship_action_string(chosen_action)
                    ship.next_action = ship_action_string(chosen_action)
        except Exception as e:
            if debugger:
                debugger.write(f"Error occurred during prediction\n")
                debugger.write(f"Error type: {type(e).__name__}\n")
                debugger.write(f"Error details: {str(e)}\n")
                debugger.write(
                    f"Input shapes - State matrices: {state_matrices.shape}, Global features: {global_features.shape}\n"
                )
        if debugger:
            debugger.write(f"Actions: {actions}\n")
        next_obs, reward = simulate_step(obs, config, actions, debugger=debugger)

        # Accumulate discounted reward
        total_reward += reward * (discount**step)
        sim_obs = next_obs

        # Early termination if ship is lost
        if ship_id not in sim_obs.players[sim_obs.player].ships:
            break

    return total_reward


def agent(obs, config):
    """Main agent function with exploration branches"""
    global model

    # Initialize debug log file if debug is enabled
    if not hasattr(agent, "debug_file") and conf.get("debug"):
        agent.debug_file = open("agent_debug.log", "w")

    if not hasattr(agent, "model"):
        agent.model = create_agent()
        if conf.get("debug"):
            agent.debug_file.write("Model created\n")

    board = Board(obs, config)
    actions = {}
    me = board.current_player

    # Log game state if debug enabled
    if conf.get("debug"):
        agent.debug_file.write(f"\nStep: {obs['step']}, Player Halite: {me.halite}\n")
        agent.debug_file.write(
            f"Ships: {len(me.ships)}, Shipyards: {len(me.shipyards)}\n"
        )

    # Process each ship
    for ship in me.ships:
        state_matrices, global_features = obs_to_transformer_input(
            obs,
            ship.id,
            obs.player,
            actions,
            debugger=agent.debug_file if conf.get("debug") else None,
        )

        # Pass inputs as separate arguments in a dictionary
        try:
            inputs = {
                "state_matrices": state_matrices,
                "global_features": global_features,
            }
            q_values = agent.model.predict(inputs)
        except Exception as e:
            if conf.get("debug"):
                agent.debug_file.write(f"Error occurred during prediction\n")
                agent.debug_file.write(f"Error type: {type(e).__name__}\n")
                agent.debug_file.write(f"Error details: {str(e)}\n")
                agent.debug_file.write(
                    f"Input shapes - State matrices: {state_matrices.shape}, Global features: {global_features.shape}\n"
                )
            q_values = np.zeros((1, 6))  # Default to zeros if prediction fails

        # Log ship state and predictions if debug enabled
        if conf.get("debug"):
            agent.debug_file.write(f"\nShip {ship.id} at position {ship.position}:\n")
            agent.debug_file.write(f"Q-values: {q_values[0]}\n")

        # Explore branches for each possible action
        exploration_values = np.zeros(6)
        for action in range(6):
            exploration_value = explore_branch(
                obs=obs,
                config=config,
                ship_id=ship.id,
                action=action,
                depth=10,
                num_branches=3,
                debugger=agent.debug_file if conf.get("debug") else None,
            )
            exploration_values[action] = exploration_value

        # Log exploration values if debug enabled
        if conf.get("debug"):
            agent.debug_file.write(f"Exploration values: {exploration_values}\n")

        # Combine model Q-values with exploration values
        combined_values = q_values[0] + 0.3 * exploration_values
        if conf.get("debug"):
            agent.debug_file.write(f"Combined values: {combined_values}\n")

        # Select best action
        action = np.argmax(q_values[0])
        actions[ship.id] = ship_action_string(action)
        ship.next_action = ShipAction(action)
        if conf.get("debug"):
            agent.debug_file.write(f"Chosen action: {ship_action_string(action)}\n")

    # Handle shipyard actions
    for shipyard in me.shipyards:
        if conf.get("debug"):
            agent.debug_file.write(f"Shipyard {shipyard.id}: {shipyard.halite}\n")
        if me.halite >= conf.get("spawnCost") and len(me.ships) < conf.get("maxShips"):
            actions[shipyard.id] = "SPAWN"
            shipyard.next_action = ShipyardAction.SPAWN
            if conf.get("debug"):
                agent.debug_file.write(f"\nShipyard {shipyard.id}: SPAWN\n")

    # Flush the debug file to ensure all data is written if debug enabled
    if conf.get("debug"):
        agent.debug_file.flush()

    return me.next_actions


def simulate_step(obs, config, actions, debugger=None):
    """Simulates one step in the environment using the Board class"""
    board = Board(obs, config)

    # Track new positions for collision detection
    new_ship_positions = {}

    # Get first opponent ID
    opponent_id = next(player_id for player_id in board.opponents)

    opponent_actions = {}

    # For each opponent ship, use our agent to predict their action
    for ship_id, ship in board.ships.items():
        if debugger:
            debugger.write(f"SIMULATE: Processing ship {ship_id}\n")
        state_matrices, global_features = obs_to_transformer_input(
            obs, ship.id, ship.player.id, actions, debugger=debugger
        )

        inputs = {"state_matrices": state_matrices, "global_features": global_features}
        q_values = agent.model.predict(inputs)
        if debugger:
            debugger.write(f"SIMULATE: Q-values: {q_values[0]}\n")
        chosen_action = np.argmax(q_values[0])
        opponent_actions[ship_id] = ship_action_string(chosen_action)

    # Combine our actions with opponent actions
    all_actions = {**actions, **opponent_actions}

    # Apply all actions and track new positions
    for ship_id, action in all_actions.items():
        if ship_id not in board.ships:
            continue

        ship = board.ships[ship_id]
        if action == "CONVERT":
            ship.next_action = ShipAction.CONVERT
        elif action == "NORTH":
            ship.next_action = ShipAction.NORTH
        elif action == "SOUTH":
            ship.next_action = ShipAction.SOUTH
        elif action == "EAST":
            ship.next_action = ShipAction.EAST
        elif action == "WEST":
            ship.next_action = ShipAction.WEST

        new_ship_positions[ship_id] = get_next_ship_pos(ship.position, action)

    if debugger:
        debugger.write(f"SIMULATE: New ship positions: {new_ship_positions}\n")

    # Update shipyard strategy with correct parameters
    # for player_id, player in board.players.items():
    #    shipyard_strat(
    #        player, obs, board, new_ship_positions, obs.configuration.episode_steps
    #    )

    if debugger:
        debugger.write(f"SIMULATE: shipyard strategy\n")

    # Simulate the next state
    next_board = board.next()

    if debugger:
        debugger.write(f"SIMULATE: next board\n")

    # Calculate reward
    current_player = board.players[board.current_player_id]
    next_player = next_board.players[next_board.current_player_id]
    current_opponent = board.players[opponent_id]
    next_opponent = next_board.players[opponent_id]

    reward = 0

    if debugger:
        debugger.write(f"SIMULATE: reward\n")

    # Reward for collecting halite
    reward += (next_player.halite - current_player.halite) * 0.01

    # Penalty for opponent collecting halite
    reward -= (next_opponent.halite - current_opponent.halite) * 0.01

    # Reward for maintaining ships
    reward += (len(next_player.ships) - len(current_player.ships)) * 100

    # Penalty for opponent gaining ships
    reward -= (len(next_opponent.ships) - len(current_opponent.ships)) * 100

    # Reward for maintaining shipyards
    reward += (len(next_player.shipyards) - len(current_player.shipyards)) * 200

    # Penalty for opponent gaining shipyards
    reward -= (len(next_opponent.shipyards) - len(current_opponent.shipyards)) * 200

    # Penalty for losing the game
    if len(next_player.ships) == 0 and next_player.halite < 500:
        reward -= 1000

    return next_board.observation, reward


def shipyard_strat(me, obs, board, new_ship_pos, episode_steps):
    if len(me.shipyards) > 0:
        shipyard = random.choice(list(me.shipyards))
        ship_nearby = False

        if obs["players"][me._id][1][shipyard._id] in new_ship_pos:
            ship_nearby = True

        if (
            me._halite >= 500
            and not ship_nearby
            and board.step < episode_steps - 200
            and len(me.ships) < 25
        ):
            shipyard.next_action = ShipyardAction.SPAWN
        elif (
            me._halite >= 500
            and not ship_nearby
            and len(me.ships) < 10
            and board.step < episode_steps - 10
        ):
            shipyard.next_action = ShipyardAction.SPAWN
