import numpy as np
import os
import gc
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K
from model import create_model

# -------------------------
# Load your training data
# train_data.shape = (N, time_steps, features)
# y_data.shape = (N, output_dim)
# -------------------------
# train_data = ...
# y_data = ...

# Create output directory
os.makedirs("Models", exist_ok=True)

for tau in np.arange(0.1, 1.0, 0.1):
    print(f"Training QR-LSTM model for quantile tau={tau:.1f}")

    model = create_model(
        input_shape=(train_data.shape[1], train_data.shape[2]),
        tau=tau
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=1e-6
    )

    history = model.fit(
        train_data, y_data,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[reduce_lr],
        verbose=1
    )

    model.save(f"Models/tau_{tau:.1f}.keras")
    K.clear_session()
    gc.collect()
