# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import sys

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) == 0:
    sys.exit(1)
test_dir = 'Stanford-Cars-dataset/test'
train_dir = 'Stanford-Cars-dataset/train'

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  seed=123,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  seed=123,
)

# data augmentation
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

# takes in a list of images
def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
        
    return images

train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

model = Sequential([
    # input shape = height, width, color channels (3 for rgb)
  layers.Rescaling(1./255, input_shape=(256, 256, 3)),
    # 16 filters (each detects a feature), 
    # relu activation (function decides if the cell on feature map has the feature)
    # kernal size = 3x3
    # each filter outputs its own feature map. 
    # the output value in the featuremap captures how strongly that feature was activated
    # each cell corresponds to a neuron and its activation value
  layers.Conv2D(16, 3, padding='same', activation='relu'),
    # max pooling to extract the max from each 2x2 portion (default size)
    # for dimensionality reduction
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # flatten from several filters/featuremaps to a flat array of neurons
  # takes each featuremap (2d or 3d) and converts to single array. concatenates all arrays to one big one
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(196)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs=3
history = model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=epochs
)

#test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
