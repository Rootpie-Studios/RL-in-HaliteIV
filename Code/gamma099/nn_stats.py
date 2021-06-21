import tensorflow as tf
import gamma099.conf as conf

ship_model = tf.keras.models.load_model(conf.get('SHIP_MODEL'))

print(ship_model.summary())