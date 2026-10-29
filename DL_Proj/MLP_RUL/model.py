"""
model.py

Multi-Layer Perceptron (MLP)
for Remaining Useful Life (RUL) Prediction
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_model(input_dim):
    """
    Creates and returns the MLP model.

    Parameters
    ----------
    input_dim : int
        Number of input features.

    Returns
    -------
    model : keras.Model
    """

    model = Sequential()

    model.add(
        Input(shape=(input_dim,))
    )

    model.add(
        Dense(
            256,
            activation="relu"
        )
    )

    model.add(
        Dropout(0.30)
    )

    # -----------------------------
    # Hidden Layer 1
    # -----------------------------
    model.add(
        Dense(
            128,
            activation='relu'
        )
    )

    model.add(
        Dropout(0.30)
    )

    # -----------------------------
    # Hidden Layer 2
    # -----------------------------
    model.add(
        Dense(
            64,
            activation='relu'
        )
    )

    # -----------------------------
    # Hidden Layer 3
    # -----------------------------
    model.add(
        Dense(
            32,
            activation='relu'
        )
    )

    # -----------------------------
    # Output Layer
    # -----------------------------
    model.add(
        Dense(
            1,
            activation='linear'
        )
    )

    # -----------------------------
    # Compile
    # -----------------------------
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


if __name__ == "__main__":

    INPUT_DIM = 570

    model = build_model(INPUT_DIM)

    model.summary()
