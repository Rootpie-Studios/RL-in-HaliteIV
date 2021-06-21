import tensorflow as tf
import se_sm_de_kp_ez_te.conf as conf

ship_model = tf.keras.models.load_model(conf.get('SHIP_MODEL'))

print(ship_model.summary())