import os
import csv
import urllib.request
import numpy as np

from src.logging import log

QUICKDRAW_NUMPY_BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

def load_class_names(classes_file_location):
  """ Loads the classes into a python array from the supplied file location """
  return open(classes_file_location).read().splitlines()

def download_examples_for_class(class_name, temp_dir):
  """ Downloads the quickdraw data for the supplied class_name """
  class_url = class_name.replace('_', '%20')
  download_url = QUICKDRAW_NUMPY_BASE_URL + f'{class_url}.npy'
  download_filepath = os.path.join(temp_dir, f'{class_name}.npy')
  file_already_exists = os.path.isfile(download_filepath)

  if (not file_already_exists):
    log(f'Downloading [{class_name}] training data from "{download_url}"')
    urllib.request.urlretrieve(download_url, download_filepath)
  else:
    log(f'Data file for [{class_name}] already exists. Using existing file.')

  return download_filepath

def load_examples_for_class(class_name, examples_dir, mmap_mode='r'):
  """
  Loads the quickdraw training data for the supplied class_name into a numpy array in mmap mode

  The data will not be loaded into memory, instead just reading from disk which allows reading
  a smal set of examples without loading all the examples into memory
  """
  examples_filepath = os.path.join(examples_dir, f'{class_name}.npy')
  return np.load(examples_filepath, mmap_mode=mmap_mode)

def process_examples(examples, image_width):
  '''
  Processes a raw array of example into a format ready for training
  
  Converts a flat array of pixels into a 2D array, and normalises the pixels values to be
  floats between 0 and 1
  '''
  # Images are 28 x 28 pixels, with each pixel value between 0 and 255
  # Reshape each example into a 2-D array image, with pixel values between 0 and 1
  examples = examples.reshape(examples.shape[0], image_width, image_width, 1).astype('float32')
  examples /= 255.0

  return examples
