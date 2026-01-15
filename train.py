import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# SETTINGS
# --------------------------
IMAGE_SIZE = (64, 64)      # Image size
BATCH_SIZE = 32
DATA_DIR = r"D:\chat_reco\dataset\augmented_images1"

# --------------------------
# LOAD DATASET
# --------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Total classes:", num_classes)

# --------------------------
# NORMALIZE DATA
# --------------------------
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------------------------------
# CNN + ANN MODEL (This is the ONLY modified section)
# -----------------------------------------------------
model = models.Sequential([

    # ----- BASIC CNN FEATURE EXTRACTOR -----
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    # flatten features → feed ANN
    layers.Flatten(),

    # ----- ANN CLASSIFIER (FULL MLP) -----
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),

    # final output layer
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --------------------------
# TRAIN
# --------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# --------------------------
# SAVE MODEL
# --------------------------
model.save("character_cnn_model.h5")
print("Model saved as character_cnn_model.h5")

# -------------------------------------------------
# BELOW THIS LINE IS ONLY THE ADDED PLOT + METRICS
# -------------------------------------------------

# Plot training/validation accuracy
plt.figure(figsize=(7,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Plot training/validation loss
plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Print average accuracy & misclassification
train_acc = np.mean(history.history['accuracy']) * 100
val_acc = np.mean(history.history['val_accuracy']) * 100
train_miss = 100 - train_acc
val_miss = 100 - val_acc

print("\n======================")
print("   FINAL STATISTICS")
print("======================")
print(f"Average Training Accuracy: {train_acc:.2f}%")
print(f"Average Validation Accuracy: {val_acc:.2f}%")
print(f"Avg Training Misclassified: {train_miss:.2f}%")
print(f"Avg Validation Misclassified: {val_miss:.2f}%")
print("======================\n")
