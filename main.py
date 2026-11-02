import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import json

print("EMNIST Balanced letöltése...")
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train[:90%]', 'train[90%:]', 'test'],
    as_supervised=True,
    with_info=True,
    shuffle_files=True
)

num_classes = ds_info.features['label'].num_classes
print(f"Osztályok száma: {num_classes}")

LABEL_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}

def preprocess(image, label):
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

BATCH_SIZE = 128
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (ds_train
            .map(preprocess, num_parallel_calls=AUTOTUNE)
            .cache()
            .shuffle(10000)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

val_ds = (ds_val
          .map(preprocess, num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))

test_ds = (ds_test
           .map(preprocess, num_parallel_calls=AUTOTUNE)
           .batch(BATCH_SIZE)
           .prefetch(AUTOTUNE))

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nTanítás indul...")
history = model.fit(
    train_ds,
    epochs=6,
    validation_data=val_ds,
    verbose=1
)

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTeszt pontosság: {test_acc:.4f}")

model.save("emnist_balanced_model.keras")
print("\nModell elmentve: emnist_balanced_model.keras")

with open("label_map.json", "w") as f:
    json.dump(LABEL_MAP, f)
print("Label map elmentve: label_map.json")