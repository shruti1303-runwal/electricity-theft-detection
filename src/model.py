"""
Wide & Deep neural network for electricity-theft detection.

Inspired by Zheng et al., "Wide and Deep CNNs for Electricity-Theft Detection"
(IEEE Trans. Industrial Informatics, 2018), which introduced the SGCC dataset.

Three input branches:
  * WIDE  : the full daily series -> Dense layers   (global patterns)
  * DEEP  : weeks x 7 "image"     -> Conv2D layers  (weekly periodicity; thieves break it)
  * STATS : engineered features    -> Dense layer    (domain knowledge)
The branches are concatenated and a sigmoid outputs P(theft).
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model(n_days: int, n_weeks: int, n_feats: int,
                units: int = 64, filters: int = 32, dropout: float = 0.3,
                lr: float = 1e-3) -> Model:
    daily_in = layers.Input(shape=(n_days,), name="daily")
    weekly_in = layers.Input(shape=(n_weeks, 7, 1), name="weekly")
    stats_in = layers.Input(shape=(n_feats,), name="stats")

    # Wide branch
    w = layers.Dense(units, activation="relu")(daily_in)
    w = layers.Dropout(dropout)(w)

    # Deep (CNN) branch
    d = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(weekly_in)
    d = layers.BatchNormalization()(d)
    d = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(d)
    d = layers.MaxPooling2D((2, 1))(d)
    d = layers.Conv2D(filters * 2, (3, 3), padding="same", activation="relu")(d)
    d = layers.GlobalAveragePooling2D()(d)
    d = layers.Dropout(dropout)(d)

    # Stats branch
    s = layers.BatchNormalization()(stats_in)
    s = layers.Dense(32, activation="relu")(s)

    x = layers.Concatenate()([w, d, s])
    x = layers.Dense(units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="theft_prob")(x)

    model = Model([daily_in, weekly_in, stats_in], out, name="wide_deep_theft")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model
