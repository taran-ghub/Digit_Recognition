import tensorflow as tf


def build_model(binary=False):
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.RandomRotation(.025, fill_mode='constant', seed=42),
        layers.RandomTranslation(.08, .08, fill_mode='constant', seed=43),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(), layers.Flatten(),
        layers.Dense(128, activation='relu'), layers.Dropout(.3),
        layers.Dense(1 if binary else 10, activation='sigmoid' if binary else 'softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),
                  loss='binary_crossentropy' if binary else 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_character_model():
    """Classify 10 digits plus 26 case-merged letters from uint8 inputs."""
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Rescaling(1./255),
        layers.RandomRotation(.025, fill_mode='constant', seed=52),
        layers.RandomTranslation(.08, .08, fill_mode='constant', seed=53),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(), layers.Flatten(),
        layers.Dense(128, activation='relu'), layers.Dropout(.3),
        layers.Dense(36, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_sequence_model():
    """CRNN logits for a 48 x 192 number image; class 10 is the CTC blank."""
    layers = tf.keras.layers
    image = layers.Input((48, 192, 1), name='number_image')
    x = layers.RandomRotation(.03, fill_mode='constant', seed=62)(image)
    x = layers.RandomTranslation(.05, .03, fill_mode='constant', seed=63)(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 1))(x)  # retain horizontal time steps
    # shape: batch, 6, 48, 128 -> 48 time steps with vertical features flattened
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((48, 6 * 128))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=.15))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=.15))(x)
    logits = layers.Dense(11, name='ctc_logits')(x)
    return tf.keras.Model(image, logits, name='number_crnn_ctc')
