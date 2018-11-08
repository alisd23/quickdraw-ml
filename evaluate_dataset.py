import numpy as np
import json
from keras.models import load_model

from src.utils import load_class_names

class_names = load_class_names('classes.txt')
counts = [None] * len(class_names)

for i, c in enumerate(class_names):
  examples = np.load(f'examples/{c}.npy', mmap_mode='r')
  counts[i] = { 'class': c, 'count': len(examples) }
  print(f'{c}: {len(examples)}')

max_class = max(counts, key=lambda c: c['count'])
min_class = min(counts, key=lambda c: c['count'])

print('Largest dataset:')
print(max_class)

print('Smallest dataset')
print(min_class)
