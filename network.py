from keras import Input, Model
from keras.layers import Dense, concatenate, BatchNormalization
from keras.optimizers import Adam, RMSprop


def build_params_model(num_hyperparameters):
    i = Input(shape=(num_hyperparameters,), name="params")
    x = Dense(64)(i)
    # x = Dense(32)(x)
    o = Dense(1)(x)
    model = Model(inputs=[i], outputs=[o])
    optimizer = RMSprop(lr=0.01)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def build_model(num_hyperparameters, num_meta_features):
    i1 = Input(shape=(num_meta_features,), name='metas')
    i2 = Input(shape=(num_hyperparameters,), name='params')
    x1 = Dense(32, activation="linear")(i1)
    x2 = Dense(32, activation="linear")(i2)
    param_out = Dense(1, activation="linear", name="param_output")(x2)
    x = concatenate([x1, x2])
    x = Dense(32, activation="linear")(x)
    x = Dense(32, activation="linear")(x)
    # x = Dense(32, activation="relu")(x)
    x = Dense(1, name="main_output", activation="linear")(x)

    model = Model(inputs=[i1, i2], outputs=[x, param_out])

    optimizer = Adam(lr=0.1)

    model.compile(optimizer=optimizer, loss="mse")
    return model
