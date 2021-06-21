import tensorflow as tf
import baseline_increment.conf as conf

ship_model = tf.keras.models.load_model(conf.get('SHIP_MODEL'))

print(ship_model.summary())