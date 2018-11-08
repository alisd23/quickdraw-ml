import numpy as np
import tensorflow.keras as keras 

from src.logging import progress, log
from src.utils import load_examples_for_class, process_examples

class DataGenerator(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(
    self, available_ids, class_names, examples_per_class, examples_dir,
    image_width=28, batch_size=32):
    'Initialization'
    self.available_ids = available_ids
    self.class_names = class_names
    self.examples_per_class = examples_per_class
    self.image_width = image_width
    self.batch_size = batch_size # Assumes each class has at exactly 'examples_per_class' examples available

    # Loop through every class, and create an mmap numpy array, referencing each class training set
    # directly from disk instead of
    log('Loading examples')
    self.examples_by_class = np.empty(len(class_names), dtype=object)
    for i, class_name in enumerate(progress(class_names)):
      self.examples_by_class[i] = load_examples_for_class(class_name, examples_dir, mmap_mode='r')
    
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.available_ids) / self.batch_size))

  def __getitem__(self, batch_index):
    'Generate one batch of data for the supplied batch index'
    # Generate ids of the batch (each id is just the index of the example if the examples were
    # concatenated into one giant array)
    batch_start = batch_index * self.batch_size
    batch_end = batch_start + self.batch_size
    batch_indexes = self.indexes[batch_start:batch_end]
    batch_ids = [self.available_ids[i] for i in batch_indexes]

    # Generate data
    x, y = self.__data_generation(batch_ids)

    return x, y

  def on_epoch_end(self):
    ''''
    Updates indexes after each epoch
    
    Randomly shuffles all the integers between 0 and number_of_examples, where each value
    represents a synthetic 'ID' of a single training example, where the ID is the index of the example
    if the examples for every class were concatenated together in one massive array.
    '''
    self.indexes = np.arange(len(self.available_ids))
    np.random.shuffle(self.indexes)

  def __data_generation(self, ids):
    'Generates data containing batch_size samples'
    # Initialization
    x = np.empty([self.batch_size, self.image_width * self.image_width], dtype=int)
    y = np.empty([self.batch_size], dtype=int)

    # Generate data
    for index, id in enumerate(ids):
      class_index = int(np.floor(id / self.examples_per_class))
      example_index = int(np.mod(id, self.examples_per_class))
      example = self.examples_by_class[class_index][example_index]
      # Store example data
      x[index] = example
      # Store example label
      y[index] = class_index

    x = process_examples(x, self.image_width)
    y = keras.utils.to_categorical(y, num_classes=len(self.class_names))

    return x, y 