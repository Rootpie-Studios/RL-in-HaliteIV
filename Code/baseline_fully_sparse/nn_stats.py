import tensorflow as tf
import baseline_fully_sparse.conf as conf

ship_model = tf.keras.models.load_model(conf.get('SHIP_MODEL'))

print(ship_model.summary())