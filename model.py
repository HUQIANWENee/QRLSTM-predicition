
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Pinball loss for quantile regression
def pinball_loss(y_true, y_pred, tau=0.5):
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error))

# QR-LSTM Model
def create_model(input_shape, tau):
    inputs = Input(shape=input_shape)

    x = LSTM(units=128, return_sequences=True, recurrent_dropout=0.1)(inputs)
    x = LSTM(units=64, return_sequences=False, recurrent_dropout=0.1)(x)
    outputs = Dense(units=24, activation=None)(x)

    # custom loss
    loss_fn = lambda y_true, y_pred: pinball_loss(y_true, y_pred, tau)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-2), loss=loss_fn)

    return model
