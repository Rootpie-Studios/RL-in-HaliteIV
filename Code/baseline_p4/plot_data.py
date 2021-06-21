from matplotlib.pyplot import title
import tensorflow as tf

import src.plot as plot
import baseline_p4.conf as conf

user_choice = input('Plot exploit data? y/n \n')

if user_choice == 'y':
    folder = conf.get('EXPLOIT_GAMES_FOLDER')
else:
    folder = conf.get('GAMES_FOLDER')


try:
    model = tf.keras.models.load_model(conf.get('SHIP_MODEL'))
except:
    model = conf.get('build_model')()
    model.save(conf.get('SHIP_MODEL'), save_format='tf')

plot.plot_progress(10, folder, conf.get('NAME'), conf.get('AGENT2')[:-3], model, conf.get('input_data'), title='baseline_p4 vs p4')