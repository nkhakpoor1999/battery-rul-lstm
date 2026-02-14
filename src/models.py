from __future__ import annotations
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model

def build_rul_model_time(W: int, F: int, l2_reg: float = 1e-4) -> Model:
    inp = layers.Input(shape=(W, F))
    x = layers.LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(inp)
    x = layers.LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    out = layers.Dense(1, activation="linear")(x)

    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
