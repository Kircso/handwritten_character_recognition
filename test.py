import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import json

model = tf.keras.models.load_model("emnist_balanced_model.keras")
print("Modell betöltve!")

with open("label_map.json", "r") as f:
    LABEL_MAP = json.load(f)
LABEL_MAP = {int(k): v for k, v in LABEL_MAP.items()}

ds_test = tfds.load('emnist/balanced', split='test', as_supervised=True)

def preprocess(image, label):
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

test_ds = ds_test.shuffle(10000).map(preprocess).batch(1).take(9)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
axes = axes.ravel()

print("Tesztelés folyamatban...")

for i, (img, true_label) in enumerate(test_ds):
    img_np = img.numpy()[0]
    true_idx = true_label.numpy()[0]
    
    pred = model.predict(img, verbose=0)
    pred_idx = np.argmax(pred)
    
    true_char = LABEL_MAP.get(true_idx, "?")
    pred_char = LABEL_MAP.get(pred_idx, "?")
    
    axes[i].imshow(img_np.squeeze(), cmap='gray')
    
    color = 'green' if true_idx == pred_idx else 'red'
    axes[i].set_title(f"Valódi: {true_char}  |  Tipp: {pred_char}", 
                      color=color, fontsize=11)
    axes[i].axis('off')

plt.suptitle("EMNIST Balanced – Szám + Betű felismerés", fontsize=14)
plt.tight_layout()
plt.show()