def get(param):
  params = {
    'AGENT1' : 'agent.py',
    'AGENT2' : 'place26_epsilon.py',
    'AGENT2_EPSILON' : 0.15,
    'AGENT1_EXPLOIT' : 'exploit.py',
    'SHIP_MODEL' : 'baseline_model',

    'SESSIONS' : 500,
    'GAMES' : 1,

    'GAMES_FOLDER' : 'games',
    'HTML_FOLDER': 'html_games',
    'EXPLOIT_GAMES_FOLDER': 'exploit_games',
    'EXPLOIT_HTML_FOLDER': 'exploit_html_games',
    
    'episodeSteps':400,
    'size' : 21,
    'epsilon' : 0.2,
    'min_epsilon' : 0.1, 
    'epsilon_decay_rate' : 0.001, # plays 400 games then caps at 0.1
    'epsilon_decay' : False,
    'PLAY_EXPLOIT' : True,

    'focus_player' : False
    }

  return params[param]