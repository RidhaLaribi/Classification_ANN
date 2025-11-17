import tensorflow as tf
import keras
from keras import layers, models
import matplotlib.pyplot as plt


# CONFIG

DATA_DIR = "../dataset/dog-cat-full-dataset/data"
IMG_SIZE = (64, 64)
BATCH = 32
EPOCHS = 15


# LOAD DATASETS

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR + "/train",
    image_size=IMG_SIZE,
    batch_size=BATCH
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR + "/test",
    image_size=IMG_SIZE,
    batch_size=BATCH
)


# NORMALIZATION

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# MODEL (ANN)

model = models.Sequential([
    layers.Flatten(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# TRAINING

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
)

model.save("dog_cat_rna_basic2.h5")



# PLOTS (Accuracy + Loss)

plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("rna_training_history.png")
plt.show()


# FINAL EVALUATION

print("\n=== FINAL EVALUATION ===")
loss, accuracy= model.evaluate(val_ds)
print(f"Accuracy: {accuracy:.4f}")

