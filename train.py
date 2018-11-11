import sys
import os
import numpy as np
import argparse
import boto3
from tensorflow.keras import callbacks
from botocore.client import ClientError

from src.logging import progress, log, title, TrainingLogger
from src.utils import load_class_names
from src.data_generator import DataGenerator
from src.model import create_model

# Set up command lin arguments for training

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument(
  '--workers',
  type=int,
  help='Number of workers to use',
  default=0
)
parser.add_argument(
  '--examples-per-class',
  dest='examples',
  type=int,
  help='Number of examples to use for each class in the full training set',
  default=1000
)
parser.add_argument(
  '--no-save',
  dest='save',
  action='store_false',
  help='If set the model/parameters are not saved',
  default=True
)

args = parser.parse_args()

title('Arguments')
log(f'Number of workers: {args.workers}')
log(f'Examples per class: {args.examples}')
log(f'Will save/upload model: {args.save}')

# Constants
EXAMPLES_DIR = os.path.abspath('examples')
EXAMPLES_PER_CLASS = args.examples
WORKERS = args.workers
IMAGE_WIDTH = 28
VALIDATION_SET_RATIO = 0.15
TEST_SET_RATIO = 0.15
BATCH_SIZE = 64
S3_MODEL_KEY = 'quickdraw.model.h5'
BUCKET_NAME = 'quickdraw-battle'

# AWS initialisation / auth checks
title('Checking AWS bucket access')
s3 = boto3.resource('s3')
bucket = s3.Bucket(BUCKET_NAME)

try:
  s3.meta.client.head_bucket(Bucket=bucket.name)
except ClientError:
  log('ERROR: Current user does not have access to AWS bucket')
  exit(0)

log(f'Access granted for current user for S3 bucket: {BUCKET_NAME}')
class_names = load_class_names('classes.txt')
dataset_size = len(class_names) * EXAMPLES_PER_CLASS

# Parameters
params = {
  'class_names': class_names,
  'examples_per_class': EXAMPLES_PER_CLASS,
  'examples_dir': EXAMPLES_DIR,
  'image_width': IMAGE_WIDTH,
  'batch_size': BATCH_SIZE,
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

title('Initialisation')
log(f'TRAIN set size: {len(train_ids)}')
log(f'VALIDATION set size: {len(validation_ids)}')
log(f'TEST set size: {len(test_ids)}\n')

# Data Generators
training_generator = DataGenerator(available_ids=train_ids, **params)
validation_generator = DataGenerator(available_ids=validation_ids, **params)
test_generator = DataGenerator(available_ids=test_ids, **params)

# Create keras model
model = create_model(len(class_names))
epochs_count = len(training_generator)
training_logger = TrainingLogger(epochs_count)

# Print model architecture
print(model.summary())
sys.stdout.flush()

# Train model on dataset
title('Training Model')
model.fit_generator(
  generator=training_generator,
  validation_data=validation_generator,
  verbose=2,
  use_multiprocessing=(WORKERS > 0),
  workers=WORKERS,
  callbacks=[callbacks.LambdaCallback(
    on_train_begin=lambda logs: training_logger.load(),
    on_batch_end=lambda batch, logs: training_logger.update(batch, logs)
  )]
)

# Evaluate on test set
title('Testing Model')
score = model.evaluate_generator(
  generator=test_generator,
  verbose=0
)

log('Top 1 test accuracy: {:0.2f}%'.format(score[1] * 100))
log('Top 5 test accuracy: {:0.2f}%'.format(score[2] * 100))

if not args.save:
  exit(0)

title('Saving Model')

# Save file locally first
log(f'Saving to local file: [{S3_MODEL_KEY}]')
model.save(S3_MODEL_KEY)
# Upload to S3
s3_client = s3.meta.client
log(f'Uploading to S3 at: [{BUCKET_NAME}:{S3_MODEL_KEY}]')
s3_client.upload_file(
  Filename=S3_MODEL_KEY,
  Bucket=BUCKET_NAME,
  Key=S3_MODEL_KEY
)