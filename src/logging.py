import sys
from tqdm import tqdm

TITLE_WIDTH = 60

def log(message, level=0):
  line_start = '=' * (level + 2)
  print(f'{line_start}  {message}')
  sys.stdout.flush()

def progress(iterable=None, total=None):
  return tqdm(iterable, total=total, ncols=80)

def title(message):
  print()
  print('=' * TITLE_WIDTH)
  print('|' + f'  {message}  '.center(TITLE_WIDTH - 2, ' ') + '|')
  print('=' * TITLE_WIDTH)
  print()
  sys.stdout.flush()

class TrainingLogger:
  def __init__(self, total=100):
    self.total = total

  def load(self):
    self.progess_bar = progress(iterable=None, total=self.total)
  
  def update(self):
    self.progess_bar.update()
