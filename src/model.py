import tensorflow as tf
from tensorflow.keras import layers, Sequential, metrics

def top_1_accuracy(y_true, y_pred):
  return metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)

def top_5_accuracy(y_true, y_pred):
  return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

def create_model(num_classes):
  input_shape = [28, 28, 1]
  model = Sequential()

  # Convolution layer 1: 28 3x3 filters, input=[28, 28, 1], output=[26, 26, 28]
  model.add(layers.Conv2D(32, kernel_size=3, input_shape=input_shape, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Flatten())
  # First fully connected layer, 1D array of [1028]
  model.add(layers.Dense(1028, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))
  # Final fully connected layer into 1D array of class scores - 1 element per class prediction 
  model.add(layers.Dense(num_classes, activation='softmax'))

  model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[top_1_accuracy, top_5_accuracy]
  )

  return model
