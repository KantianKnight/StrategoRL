# Q-learning with Function Approximation

import numpy as np
from stratego_env.examples.util import softmax
from stratego_env import StrategoMultiAgentEnv, ObservationComponents, ObservationModes, GameVersions

from keras.models import Model, load_model
from keras.layers import Input, Dense, Concatenate, Dropout, LeakyReLU

from QL_features import get_state_features, get_action_features, convert_index, convert_captured_layer


### ------------------------ ###
# Create Function Approximator #
### ------------------------ ###

def create_q_network(own_shape, opp_shape, captured_shape, unmoved_shape, action_shape, piece_shape):
    own_input = Input(shape=own_shape, name='own_input')
    opp_input = Input(shape=opp_shape, name='opp_input')
    captured_input = Input(shape=captured_shape, name='captured_input')
    unmoved_input = Input(shape=unmoved_shape, name='unmoved_input')
    action_input = Input(shape=action_shape, name='action_input')
    piece_input = Input(shape=piece_shape, name='piece_input')

    # Feedforward Layers for Own Input
    own_dense = Dense(64, activation='relu')(own_input)
    own_dropout = Dropout(0.2)(own_dense)

    # Feedforward Layers for Opp Input
    opp_dense = Dense(64, activation='relu')(opp_input)
    opp_dropout = Dropout(0.2)(opp_dense)

    # Feedforward Layers for Captured Input
    captured_dense = Dense(32, activation='relu')(captured_input)
    captured_dropout = Dropout(0.2)(captured_dense)

    # Feedforward Layers for Unmoved Input
    unmoved_dense = Dense(64, activation='relu')(unmoved_input)
    unmoved_dropout = Dropout(0.2)(unmoved_dense)

    # Feedforward Layers for Action Input
    action_dense = Dense(128, activation='relu')(action_input)
    action_dropout = Dropout(0.2)(action_dense)

    # Feedforward Layers for Piece Input
    piece_dense = Dense(32, activation='relu')(piece_input)
    piece_dropout = Dropout(0.2)(piece_dense)

    # Concatenation Layer
    concat = Concatenate()([own_dropout, opp_dropout, captured_dropout, unmoved_dropout, action_dropout, piece_dropout])

    # Dense Layers
    dense1 = Dense(192, activation='relu')(concat)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(96, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    dense3 = Dense(48, activation='relu')(dropout2)
    q_value = Dense(1, name='q_value')(dense3)

    # Model Compilation
    q_network = Model(inputs=[own_input, opp_input, captured_input, unmoved_input, action_input, piece_input], outputs=q_value)
    q_network.compile(optimizer='adam', loss='mse')

    return q_network

# Define Model Shapes
own_shape = (100,)
opp_shape = (48,)
captured_shape = (6,)
unmoved_shape = (8,)
action_shape = (64,)
piece_shape = (3,)

# Create the Model
model = create_q_network(own_shape, opp_shape, captured_shape, unmoved_shape, action_shape, piece_shape)
print(model.summary())

# For Loading and Saving the Model
iter = 0

if iter > 0:
    model = load_model("Models/Current/QLTiny" + str(iter) + ".h5")
    print(model.summary())


### ---------------------- ###
# Initialize the Environment #
### ---------------------- ###

config = {
    'version': GameVersions.TINY,
    'random_player_assignment': True,
    'human_inits': True,
    'penalize_ties': True,
    'observation_mode': ObservationModes.PARTIALLY_OBSERVABLE,
    # 'observation_mode': ObservationModes.BOTH_OBSERVATIONS,
}

env = StrategoMultiAgentEnv(env_config=config)


### ---------------------- ###
# Define the Hyperparameters #
### ---------------------- ###

alpha = 0.2          # learning rate
gamma = 0.95         # discount factor
epsilon = 0.1        # exploration rate
num_episodes = 2000  # number of episodes to train for

# Track Score
win_count = 0
draw_count = 0
loss_count = 0


### ------------------ ###
# Iterate over the games #
### ------------------ ###

for i in range(100*iter + 1, 100*iter + 1 + num_episodes):
    # print("Iteration", i)

    # Reset the environment
    obs = env.reset()
    assert len(obs.keys()) == 1
    current_player = list(obs.keys())[0]
    assert current_player == 1 or current_player == -1
    # Note: Agent Bot is 1, Random Bot is -1

    # Archive for past states
    previous_state = np.array([])
    previous_action_space = np.array([])
    previous_action_index = -1
    previous_q_value = 0

    # Initial state
    current_state = obs[current_player][ObservationComponents.PARTIAL_OBSERVATION.value]

    # Initialize action space
    action_space = obs[current_player][ObservationComponents.VALID_ACTIONS_MASK.value]
    current_action_space = action_space[:, :, [0, 3, 6, 9]]

    # Loop over the moves in the game
    while True:
        # Get valid moves
        flattened_action_space = current_action_space.flatten()
        valid_indices = np.where(flattened_action_space == 1)[0]
        # print(valid_indices)

        # If random bot
        if current_player == -1:
            valid_index = np.random.choice(range(len(valid_indices)))
            action_index = valid_indices[valid_index]
        
        # If custom agent
        else:
            ### -------------------- ###
            # Evaluate potential moves #
            ### -------------------- ###

            valid_count = len(valid_indices)

            own_pieces, opp_pieces, captured_pieces, unmoved_pieces = get_state_features(current_state)

            tiled_features = []
            for feature in [own_pieces, opp_pieces, captured_pieces, unmoved_pieces]:
                tiled_feature = np.tile(feature, valid_count).reshape(valid_count, -1)
                tiled_features.append(tiled_feature)
            own_pieces, opp_pieces, captured_pieces, unmoved_pieces = tiled_features
            

            possible_actions_list = []
            possible_piece_type_list = []
            for possible_action_index in valid_indices:
                possible_action_features, possible_piece_type = get_action_features(current_state, possible_action_index)
                possible_actions_list.append(possible_action_features)
                possible_piece_type_list.append(possible_piece_type)
            
            action_features = np.array(possible_actions_list)
            piece_type = np.array(possible_piece_type_list)


            if len(action_features) == 0:
                possible_q_values = [0]
            else:
                possible_q_values = model.predict(([own_pieces, opp_pieces, captured_pieces, unmoved_pieces, action_features, piece_type]), verbose=0)

            # print(possible_q_values)
            

            ### ----------------------- ###
            # Train Function Approximator #
            ### ----------------------- ###

            # Ensure it's not the first move
            if previous_action_index != -1:
                # Calculate the maximum Q-value for the next state
                max_possible_q_value = np.max(possible_q_values)

                # Add a piece bias
                piece_bias = 0

                piece_indices = [4, 5]
                for j in piece_indices:
                    piece_bias -= (j-3) * np.sum(convert_captured_layer(current_state[:, :, j+41]))
                    piece_bias += (j-3) * np.sum(convert_captured_layer(current_state[:, :, j+53]))
                # print(piece_bias)

                # Calculate Q-value target
                q_learning_target = (reward[current_player] * 10) + (gamma * max_possible_q_value) + piece_bias

                # Update the Q-value approximation
                updated_q_value = previous_q_value + alpha * (q_learning_target - previous_q_value)
                # print(updated_q_value)

                # Train the function approximator
                previous_own_pieces, previous_opp_pieces, previous_captured_pieces, previous_unmoved_pieces = get_state_features(previous_state)
                previous_own_pieces = np.array([previous_own_pieces])
                previous_opp_pieces = np.array([previous_opp_pieces])
                previous_captured_pieces = np.array([previous_captured_pieces])
                previous_unmoved_pieces = np.array([previous_unmoved_pieces])
                
                previous_action_features, previous_piece_type = get_action_features(previous_state, previous_action_index)
                previous_action_features = np.array([previous_action_features])
                previous_piece_type = np.array([previous_piece_type])

                # predicted = model.predict(([previous_state_features, previous_action_features, previous_piece_type]), verbose=0)
                # print(predicted, updated_q_value)

                model.train_on_batch([previous_own_pieces, previous_opp_pieces, 
                                      previous_captured_pieces, previous_unmoved_pieces,
                                      previous_action_features, previous_piece_type], 
                                      np.array([updated_q_value]))


            ### ---------------------------- ###
            # Choose move using Epsilon-Greedy #
            ### ---------------------------- ###

            if np.random.rand() < epsilon:
                valid_index = np.random.choice(range(len(valid_indices)))
                
            else:
                valid_index = np.argmax(possible_q_values)
            
            action_index = valid_indices[valid_index]


            ### ---------------- ###
            # Overwrite old values #
            ### ---------------- ###

            previous_state = current_state
            previous_action_space = current_action_space
            previous_action_index = action_index
            previous_q_value = possible_q_values[valid_index]


        ### ------------------------------------------------ ###
        # Take the action and observe the new state and reward #
        ### ------------------------------------------------ ###

        # Take the action
        obs, reward, done, info = env.step(action_dict={current_player: convert_index(action_index)})
        
        # Swap player
        current_player = -current_player

        # Update state, action space, and action index for next iteration
        current_state = obs[current_player][ObservationComponents.PARTIAL_OBSERVATION.value]

        action_space = obs[current_player][ObservationComponents.VALID_ACTIONS_MASK.value]
        current_action_space = action_space[:, :, [0, 3, 6, 9]]

        # Track stats
        if reward[current_player] == current_player:
            win_count += 1
        
        elif reward[current_player] == -current_player:
            loss_count += 1

        elif reward[current_player] != -1 and reward[current_player] != 0 and reward[current_player] != 1:
            draw_count += 1

        # Break if game is over
        if(done["__all__"]):
            break

    # if i%50 == 0:
    #     print("WDL:", win_count, draw_count, loss_count)
    #     win_count = 0
    #     draw_count = 0
    #     loss_count = 0

    # if i%100 == 0:
    #     print("Iteration", i)
    #     model.save("Models/QLFA" + str(i//100) + ".h5")

# Old: Single Model Save
# model.save('QLFA.h5')