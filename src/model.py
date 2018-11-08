import tensorflow as tf
from tensorflow.keras import layers, Sequential, metrics

def top_1_accuracy(y_true, y_pred):
  return metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)

def top_5_accuracy(y_true, y_pred):
  return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

def create_model(image_width, num_classes, image_channels=1):
  model = Sequential()
  # Convolution layer 1: 16 3x3 filters, padding calculated as to generate an output shape of
  # [image_width, image_width, 16] => [X, Y, Depth]
  model.add(
    layers.Convolution2D(
      16,
      (3, 3),
      padding='same',
      input_shape=[image_width, image_width, image_channels],
      activation='relu'
    )
  )
  # Pooling layer 1: output volume: [image_width / 2, image_width / 2, 16]
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  # Convolution layer 2: 32 3x3 filters, padding calculated as to generate an output shape of
  # [32, image_width / 2, image_width / 2]
  model.add(layers.Convolution2D(32, (3, 3), padding='same', activation='relu'))
  # Pooling layer 1: output volume: [image_width / 4, image_width / 4, 32]
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  # Convolution layer 3: 64 3x3 filters, padding calculated as to generate an output shape of
  # [64, image_width / 4, image_width / 4, 64]
  model.add(layers.Convolution2D(64, (3, 3), padding='same', activation='relu'))
  # Pooling layer 1: output volume: [image_width / 8, image_width / 8, 64]
  model.add(layers.MaxPooling2D(pool_size=(2,2)))
  # Flatten 2D image data to create 1D array: [(image_width / 8) * (image_width / 8) * 64]
  model.add(layers.Flatten())
  # First fully connected layer, 1D array of [128]
  model.add(layers.Dense(128, activation='relu'))
  # Final fully connected layer into 1D array of class scores - 1 element per class prediction 
  model.add(layers.Dense(num_classes, activation='softmax')) 
  # Train model
  adam = tf.keras.optimizers.Adam()

  model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=[top_1_accuracy, top_5_accuracy]
  )

  return model
