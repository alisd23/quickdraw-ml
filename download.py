import os
import csv
import numpy as np
import shutil

from src.logging import log, progress
from src.utils import download_examples_for_class, load_class_names

# The maximum number of training examples to load per class
MAX_EXAMPLES_PER_CLASS = 50000
EXAMPLES_DATA_DIR = os.path.abspath('examples')

log(f'Creating training examples directory if it necessary at: "{EXAMPLES_DATA_DIR}"')
os.makedirs(EXAMPLES_DATA_DIR, exist_ok=True)

# Get array of classes that we are training on
classes = load_class_names('classes.txt')

log('Downloading training data')

# Loop through each class, downloading the training examples and saving them
for class_index, class_name in enumerate(classes):
  print(f" ({class_index+1}/{len(classes)})", end=" ")
  download_examples_for_class(class_name, EXAMPLES_DATA_DIR)
