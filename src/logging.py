import sys
from tqdm import tqdm

def log(message, level=0):
  line_start = '=' * (level + 2)
  print(f'{line_start} {message}')
  sys.stdout.flush()

def progress(iterable=None, total=None):
  return tqdm(iterable, total=total, ncols=80)
