import tensorflow as tf
import enemy_data_team_spirit.conf as conf

ship_model = tf.keras.models.load_model(conf.get('SHIP_MODEL'))

print(ship_model.summary())