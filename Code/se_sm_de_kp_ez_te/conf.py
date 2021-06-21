from pickle import FALSE
import src.input_data as input_data
import src.model as model
import src.train_functions as train_functions

def get(param):
  method_name = 'se_sm_de_kp_ez_te/'
  params = {
    'NAME' : method_name[:-1],

    'AGENT1' : method_name + 'agent.py',
    'AGENT2' : method_name + 'self.py',
    'AGENT1_EXPLOIT' : method_name + 'exploit.py',
    'SHIP_MODEL' : method_name + 'selfplay_smaller-deeper_kill-penalty_epsilon-zero_team-reward_model',

    'SESSIONS' : 3000,
    'GAMES' : 1,

    'GAMES_FOLDER' : method_name + 'games',
    'HTML_FOLDER': method_name + 'html',
    'EXPLOIT_GAMES_FOLDER': method_name + 'exploit_games',
    'EXPLOIT_HTML_FOLDER': method_name + 'exploit_html',
    
    'episodeSteps':400,
    'size' : 21,
    'epsilon' : 0.05,
    'min_epsilon' : 0, 
    'epsilon_decay_rate' : 0.001,
    'epsilon_decay' : False,
    'PLAY_EXPLOIT' : True,

    'input_data' : input_data.obs_to_matrix_sparse,
    'build_model' : model.build_smaller_deeper_model,
    'train_function' : train_functions.train_on_games_final
    }

  return params[param]