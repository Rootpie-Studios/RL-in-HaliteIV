from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.engine.base_layer import InputSpec

def build_baseline_model(action_space=5, img_size=21, nbr_imgs=8, lr=0.0001):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), input_shape=(img_size, img_size, nbr_imgs)))
    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    
    return model 

def build_wider_model(action_space=5, img_size=21, nbr_imgs=8, lr=0.0001):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), input_shape=(img_size, img_size, nbr_imgs)))
    model.add(Flatten())

    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    
    return model 

def build_deeper_model(action_space=5, img_size=21, nbr_imgs=8, lr=0.0001):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), input_shape=(img_size, img_size, nbr_imgs)))
    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))

    return model

def build_compact6_model(action_space=5, img_size=21, nbr_imgs=6, lr=0.0001):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), input_shape=(img_size, img_size, nbr_imgs)))
    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))

    return model

def build_nearsighted_model(action_space=5, img_size=11, nbr_imgs=8, lr=0.0001):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), input_shape=(img_size, img_size, state_space)))
    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    
    return model 
    
def build_tiny_model(action_space=5, img_size=21, nbr_imgs=8, lr=0.0001):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), input_shape=(img_size, img_size, nbr_imgs)))
    model.add(Flatten())

    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    
    return model 

def build_smaller_model(action_space=5, img_size=21, nbr_imgs=8, lr=0.0001):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), input_shape=(img_size, img_size, nbr_imgs)))
    model.add(Flatten())

    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    
    return model 

def build_smaller_deeper_model(action_space=5, img_size=21, nbr_imgs=9, lr=0.0001):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), input_shape=(img_size, img_size, nbr_imgs)))
    model.add(Flatten())

    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    
    return model 