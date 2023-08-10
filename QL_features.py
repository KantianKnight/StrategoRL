import numpy as np

# Define the features based on domain knowledge
def get_state_features(state):
    # We exclude 10 <-> flag
    piece_indices = [4, 5, 6]


    ### ------------------- ###
    # Own Pieces (100 inputs) #
    ### ------------------- ###

    own = []
    # Own Pieces (Can't be labeled Unknown)
    # 3*4*4 = 48 inputs
    for i in piece_indices:
        own.append(np.array(state[:, :, i]))

    # Flag: 4 inputs
    own.append(np.array(state[:, :, 10][0]))
    
    # Own Pieces Revealed (Can be labeled Unknown)
    # 3*4*4 = 48 inputs
    for i in piece_indices:
        own.append(np.array(state[:, :, i+12]))

    own = np.concatenate([np.array(x).flatten() if isinstance(x, np.ndarray) else np.array(x).reshape(1) for x in own])


    ### ------------------ ###
    # Opp Pieces (48 inputs) #
    ### ------------------ ###

    opp = []
    # Opp Pieces Revealed
    # 3*4*4 = 48 inputs
    for i in piece_indices:
        opp.append(np.array(state[:, :, i+25]))
    
    opp = np.concatenate([np.array(x).flatten() if isinstance(x, np.ndarray) else np.array(x).reshape(1) for x in opp])
    

    ### ---------------------- ###
    # Captured Pieces (6 inputs) #
    ### ---------------------- ###

    captured = []
    for i in piece_indices:
        # Own Pieces Captured
        # 3*1 = 3 inputs
        captured.append(np.sum(convert_captured_layer(state[:, :, i+41])))

        # Opp Pieces Captured
        # 3*1 = 3 inputs
        captured.append(np.sum(convert_captured_layer(state[:, :, i+53])))


    ### --------------------- ###
    # Unmoved Pieces (8 inputs) #
    ### --------------------- ###

    unmoved = np.concatenate((state[:, :, 65][0], state[:, :, 66][3]), axis=0)


    return np.array(own).reshape(-1), np.array(opp).reshape(-1), np.array(captured).reshape(-1), np.array(unmoved).reshape(-1)


# def get_action_features(action_index):
#     ### -------------------------------- ###
#     # Action Space (log(4*4*4) = 6 inputs) #
#     ### -------------------------------- ###
    
#     # (4 channels) 0 (R), 3 (L), 6 (U), 9 (D)
#     # We can present 64 possibilities with 6 binary vals
#     bin_index = bin(action_index)[2:].zfill(6)
#     features = [int(bit) for bit in bin_index]

#     return np.array(features).reshape(-1)


def get_action_features(state, action_index):
    ### ---------------------------- ###
    # Action Space (4*4*4 = 64 inputs) #
    ### ---------------------------- ###

    # (4 channels) 0 (R), 3 (L), 6 (U), 9 (D)
    features = []
    action_mask = np.zeros((4, 4, 4))

    # One Hot Encoding 
    action_mask[action_index//16, (action_index//4)%4, action_index%4] = 1
    for i in range(4):
        features.append(action_mask[:, :, i])


    ### ----------------- ###
    # Piece Type (3 inputs) #
    ### ----------------- ###

    # 3 Features for Piece Type
    piece_type = np.array([0, 0, 0])

    # Identify Piece Type:
    piece_indices = [4, 5, 6]
    for i in piece_indices:
        if state[action_index//16, (action_index//4)%4, i] == 1:
            piece_type[i-4] = 1


    return np.array(features).reshape(-1), np.array(piece_type).reshape(-1)


# Converting index since we did [0, 3, 6, 9]
def convert_index(action_index):
    add1 = action_index//4
    add2 = action_index%4
    return action_index + (add1 * 9) + (add2 * 2)


# Since values are -1 for not captured and -0.75 for captured
def convert_captured_layer(captured_layer):
    return (captured_layer + 1) * 4