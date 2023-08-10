from stratego_env import StrategoMultiAgentEnv, ObservationComponents, ObservationModes, GameVersions
from stratego_env.examples.util import softmax
from QL_features import get_state_features, get_action_features, convert_index
from nnet import choose_random_action

import numpy as np
from keras.models import load_model

### -------- ###
# Custom Agent #
### -------- ###

def ql_choose_action(current_player, obs):
    ### ----------------- ###
    # Get board information #
    ### ----------------- ###

    # Get the state
    current_state = obs[current_player][ObservationComponents.PARTIAL_OBSERVATION.value]

    # Get the action space
    action_space = obs[current_player][ObservationComponents.VALID_ACTIONS_MASK.value]
    current_action_space = action_space[:, :, [0, 3, 6, 9]]

    # Get valid indices
    flattened_action_space = current_action_space.flatten()
    valid_indices = np.where(flattened_action_space == 1)[0]
    # print(valid_indices)


    ### ------------------ ###
    # Extract board features #
    ### ------------------ ###

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


    ### ------------------------ ###
    # Calculate potential Q-values #
    ### ------------------------ ###

    if len(action_features) == 0:
        possible_q_values = [0]
    else:
        possible_q_values = model.predict(([own_pieces, opp_pieces, captured_pieces, unmoved_pieces, action_features, piece_type]), verbose=0)
    # print(possible_q_values)


    ### ----------------- ###
    # Exploitation Gameplay #
    ### ----------------- ###

    valid_index = np.argmax(possible_q_values)
    action_index = valid_indices[valid_index]
    # print(valid_index)


    ### ------------------------ ###
    # Nash + Exploitation Gameplay #
    ### ------------------------ ###

    # # New decision tree with filtering
    # q_values = np.array(possible_q_values).flatten()
    # sorted_indices = np.argsort(q_values)
    # top_count = 2

    # # In case there are < n values
    # nth_largest = q_values[sorted_indices[max(-top_count, -len(q_values))]]

    # # Remove all values except the n largest
    # q_values[q_values < nth_largest] = np.maximum(np.log(1e-8), np.finfo(np.float32).min)

    # action_probabilities = np.array(softmax(q_values)).flatten()
    # valid_index = np.random.choice(range(len(action_probabilities)), p=action_probabilities)
    # action_index = valid_indices[valid_index]


    ### -------------- ###
    # Nash-Like Gameplay #
    ### -------------- ###

    # # Old decision tree without filtering
    # action_probabilities = np.array(softmax(np.array(q_values))).flatten()
    # action_index = valid_indices[np.random.choice(range(len(action_probabilities)), p=action_probabilities)]


    ### --------------------- ###
    # Board State for Debugging #
    ### --------------------- ###

    # full = obs[current_player][ObservationComponents.FULL_OBSERVATION.value]
    # print(full[:, :, 4]*1+full[:, :, 5]*2+full[:, :, 6]*3+full[:, :, 10]*9+
    #       full[:, :, 16]*(-1)+full[:, :, 17]*(-2)+full[:, :, 18]*(-3)+full[:, :, 22]*(-9),"\n")

    return convert_index(action_index)


if __name__ == '__main__':
    ### ------------- ###
    # Setup environment #
    ### ------------- ###

    config = {
        'version': GameVersions.TINY,
        'random_player_assignment': True,
        'human_inits': True,
        'penalize_ties': True,
        # 'observation_mode': ObservationModes.PARTIALLY_OBSERVABLE,
        'observation_mode': ObservationModes.BOTH_OBSERVATIONS,
    }

    env = StrategoMultiAgentEnv(env_config=config)


    ### -------------- ###
    # Simulate games! :D #
    ### -------------- ###

    model = load_model("QL_function_approximator.h5")

    wins, draws, losses = 0, 0, 0
    
    number_of_games = 500
    for i in range(1, number_of_games+1):
        # print("New Game Started")
        obs = env.reset()
        move_count = 0

        while True:
            assert len(obs.keys()) == 1
            current_player = list(obs.keys())[0]
            assert current_player == 1 or current_player == -1

            # Custom Agent
            if current_player == 1:
                current_player_action = ql_choose_action(current_player=current_player, obs=obs)
            
            # Default Random Bot
            else:
                current_player_action = choose_random_action(current_player=current_player, obs=obs)

            obs, rew, done, info = env.step(action_dict={current_player: current_player_action})
            # print(f"Player {current_player} made move {current_player_action}")

            if done["__all__"]:
                if rew[1] == 1:
                    wins += 1
                elif rew[1] == -1:
                    losses += 1
                else:
                    draws += 1
                # print(f"Game Finished, player 1 rew: {rew[1]}, player -1 rew: {rew[-1]}")

                break

            else:
                assert all(r == 0.0 for r in rew.values())
        
        if i == number_of_games:
            print("Game", i)
            print(wins, draws, losses)