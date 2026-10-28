import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

print("EMNIST Balanced (számok + betűk) letöltése a TFDS-en keresztül...")

# Hivatalos, stabil TFDS letöltés
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True,
)

print("Adatok előkészítése és forgatása...")
def convert_data(ds):
    images = []
    labels = []
    for img, label in tfds.as_numpy(ds):
        # EMNIST orientációs korrekció, hogy a karakterek a helyesen álljanak
        img = np.fliplr(np.rot90(img.squeeze(), 1))
        images.append(img)
        labels.append(label)
    return np.array(images) / 255.0, np.array(labels)

x_train, y_train = convert_data(ds_train)
x_test, y_test = convert_data(ds_test)

num_classes = ds_info.features['label'].num_classes
print(f"Osztályok száma: {num_classes} (számok és betűk)")

# CNN modell építése a 47 osztályhoz
model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Modell tanítása a számokra és betűkre...")
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=64)

# Mentés a helyes néven
model.save("emnist_balanced_model.keras")
print("Siker! Az EMNIST balanced modell elmentve 'emnist_balanced_model.keras' néven.")