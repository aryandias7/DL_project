"""
model.py

Transformer encoder for Remaining Useful Life (RUL) regression.

  input  (B, 30, 18)  sensor window
    -> Dense(d_model) + learned positional embedding
    -> N x [LayerNorm -> Multi-Head Self-Attention -> residual
            LayerNorm -> Feed-Forward                -> residual]
    -> LayerNorm -> [mean-pool over time || last timestep]
    -> Dense(64) -> Dense(32) -> Dense(1)  = RUL
"""
import keras
from keras import layers

import config


@keras.saving.register_keras_serializable(package="rul")
class PositionalEmbedding(layers.Layer):
    """Learned positional embedding added to every timestep.

    Self-attention ignores order, so without this the model could not tell
    the oldest cycle in the window from the newest.
    """

    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_emb = layers.Embedding(input_dim=seq_len, output_dim=d_model)

    def call(self, x):
        positions = keras.ops.arange(self.seq_len)
        return x + self.pos_emb(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return cfg


def encoder_block(x, d_model, num_heads, ff_dim, dropout):
    """Pre-LayerNorm Transformer encoder block."""
    h = layers.LayerNormalization(epsilon=1e-6)(x)
    h = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
    )(h, h)
    h = layers.Dropout(dropout)(h)
    x = layers.Add()([x, h])

    h = layers.LayerNormalization(epsilon=1e-6)(x)
    h = layers.Dense(ff_dim, activation="gelu")(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Dense(d_model)(h)
    h = layers.Dropout(dropout)(h)
    return layers.Add()([x, h])


def build_model(
    window_size,
    num_features,
    d_model=config.D_MODEL,
    num_heads=config.NUM_HEADS,
    ff_dim=config.FF_DIM,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT,
    learning_rate=config.LEARNING_RATE,
):
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    inputs = layers.Input(shape=(window_size, num_features), name="sensor_window")

    x = layers.Dense(d_model)(inputs)
    x = PositionalEmbedding(window_size, d_model)(x)
    x = layers.Dropout(dropout)(x)

    for _ in range(num_layers):
        x = encoder_block(x, d_model, num_heads, ff_dim, dropout)

    x = layers.LayerNormalization(epsilon=1e-6)(x)

    avg = layers.GlobalAveragePooling1D()(x)                                   # whole history
    last = layers.Lambda(lambda t: t[:, -1, :], output_shape=(d_model,))(x)    # latest cycle
    x = layers.Concatenate()([avg, last])

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs, outputs, name="transformer_rul")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mse",
        metrics=["mae"],
    )
    return model


if __name__ == "__main__":
    build_model(config.WINDOW_SIZE, 18).summary()
