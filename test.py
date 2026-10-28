import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. A betanított modell betöltése
model = tf.keras.models.load_model("handwritten_model.keras")
print("Modell sikeresen betöltve a teszteléshez!")

# 2. A teszt adatok betöltése
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()
x_test_normalized = x_test / 255.0

# 3. Válasszunk ki 9 darab véletlenszerű képet a teszthalmazból
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
axes = axes.ravel()

for i in range(9):
    index = np.random.randint(0, len(x_test))
    sample_image = x_test_normalized[index]
    true_label = y_test[index]

    # Predikció
    prediction = model.predict(np.expand_dims(sample_image, axis=0), verbose=0)
    predicted_label = np.argmax(prediction)

    # Megjelenítés rácsban
    axes[i].imshow(sample_image, cmap='gray')
    
    # Ha eltalálta, zöld, ha nem, piros színű a felirat
    color = 'green' if true_label == predicted_label else 'red'
    axes[i].set_title(f"Valódi: {true_label} | Tipp: {predicted_label}", color=color, fontsize=10)
    axes[i].axis('off')

plt.tight_layout()
print("Grafikus ablak megnyitása a rácsos tesztekkel...")
plt.show()