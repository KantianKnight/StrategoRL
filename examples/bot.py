from stratego_env import StrategoMultiAgentEnv, ObservationComponents, ObservationModes, GameVersions
from stratego_env.examples.util import softmax
import numpy as np

from Old.QL_featuresOld import get_features, convert_index
from keras.models import load_model

model = load_model('QLTiny.h5')

def ql_choose_action(current_player, obs):
    # Get the state and action space
    state = obs[current_player][ObservationComponents.PARTIAL_OBSERVATION.value]
    action_space = obs[current_player][ObservationComponents.VALID_ACTIONS_MASK.value]
    action_space = action_space[:, :, [0, 3, 6, 9]]

    valid_indices = np.where(action_space.flatten() == 1)[0]
    q_values = []
    for action_index in valid_indices:
        features = get_features(state, action_space, action_index)
        q_values.append(model.predict(np.array([features]), verbose=0))

    action_probabilities = np.array(softmax(np.array(q_values))).flatten()
    action_index = valid_indices[np.random.choice(range(len(action_probabilities)), p=action_probabilities)]

    return convert_index(action_index)


if __name__ == '__main__':
    config = {
        'version': GameVersions.TINY,
        'random_player_assignment': False,
        'human_inits': True,
        'observation_mode': ObservationModes.PARTIALLY_OBSERVABLE,

        'vs_human': True,  # one of the players is a human using a web gui
        'human_player_num': -1,  # 1 or -1
        'human_web_gui_port': 8500,
    }

    env = StrategoMultiAgentEnv(env_config=config)

    print(f"Visit \nhttp://localhost:{config['human_web_gui_port']}?player={config['human_player_num']} on a web browser")
    env_agent_player_num = config['human_player_num'] * -1

    number_of_games = 2
    for _ in range(number_of_games):
        print("New Game Started")
        obs = env.reset()
        while True:

            assert len(obs.keys()) == 1
            current_player = list(obs.keys())[0]
            assert current_player == env_agent_player_num

            current_player_action = ql_choose_action(current_player=current_player, obs_from_env=obs)

            obs, rew, done, info = env.step(action_dict={current_player: current_player_action})
            print(f"Player {current_player} made move {current_player_action}")

            if done["__all__"]:
                print(f"Game Finished, player {env_agent_player_num} rew: {rew[env_agent_player_num]}")
                break
            else:
                assert all(r == 0.0 for r in rew.values())