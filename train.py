import os
import numpy as np

from src.logging import progress, log
from src.utils import load_class_names
from src.data_generator import DataGenerator
from src.model import create_model

EXAMPLES_DIR = os.path.abspath('examples')
EXAMPLES_PER_CLASS = 1000
IMAGE_WIDTH = 28
VALIDATION_SET_RATIO = 0.15
TEST_SET_RATIO = 0.15
S3_MODEL_KEY = 'quickdraw.h5.model'

class_names = load_class_names('classes.txt')
dataset_size = len(class_names) * EXAMPLES_PER_CLASS

# Parameters
params = {
  'class_names': class_names,
  'examples_per_class': EXAMPLES_PER_CLASS,
  'examples_dir': EXAMPLES_DIR,
  'image_width': IMAGE_WIDTH,
  'batch_size': 64,
}

# Generate array of numbers 0 - dataset_size, to represent example IDs
all_ids = np.arange(dataset_size)
# Shuffle IDs randomly
np.random.shuffle(all_ids)

# Example ID Datasets
validation_split_point = int(dataset_size * (1 - VALIDATION_SET_RATIO - TEST_SET_RATIO))
test_split_point = int(dataset_size * (1 - TEST_SET_RATIO))

train_ids = all_ids[0:validation_split_point]
validation_ids = all_ids[validation_split_point:test_split_point]
test_ids = all_ids[test_split_point:dataset_size]

print()
log(f'TRAIN set size: {len(train_ids)}')
log(f'VALIDATION set size: {len(validation_ids)}')
log(f'TEST set size: {len(test_ids)}\n')

# Data Generators
training_generator = DataGenerator(available_ids=train_ids, enable_logging=True, **params)
validation_generator = DataGenerator(available_ids=validation_ids, **params)
test_generator = DataGenerator(available_ids=test_ids, **params)

# Design model
model = create_model(IMAGE_WIDTH, len(class_names))

print()
log('TRAINING model')
# Train model on dataset
model.fit_generator(
  generator=training_generator,
  validation_data=validation_generator,
  verbose = 2,
  # use_multiprocessing=True,
  workers=1)

print()
log('TESTING model')
# Evaluate on test set
score = model.evaluate_generator(
  generator=test_generator,
  verbose=0
)
print('\nTest accuracy: {:0.2f}%'.format(score[1] * 100))

# Save file locally first
model.save(S3_MODEL_KEY)
