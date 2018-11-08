import sys
from tqdm import tqdm

TITLE_WIDTH = 60

def log(message, level=0):
  line_start = '=' * (level + 2)
  print(f'{line_start}  {message}')
  sys.stdout.flush()

def progress(iterable=None, total=None, postfix=None, ncols=80):
  return tqdm(iterable, total=total, ncols=ncols, postfix=postfix)

def title(message):
  print()
  print('=' * TITLE_WIDTH)
  print('|' + f'  {message}  '.center(TITLE_WIDTH - 2, ' ') + '|')
  print('=' * TITLE_WIDTH)
  print()
  sys.stdout.flush()

def format_accuracy(accuracy):
  return '{:0<5.1f}%'.format(accuracy * 100)

class TrainingLogger:
  def __init__(self, total=100):
    self.total = total

  def load(self):
    stats = {
      'top_5': format_accuracy(0),
      'top_1': format_accuracy(0),
    }
    self.progress_bar = progress(iterable=None, total=self.total, postfix=stats, ncols=110)
  
  def update(self, batch, logs):
    # Only update accuracy every n batches
    if batch % 10 == 0:
      stats = {
        'top_5': format_accuracy(logs['top_5_accuracy']),
        'top_1': format_accuracy(logs['top_1_accuracy']),
      }
      self.progress_bar.set_postfix(stats, refresh=False)

    self.progress_bar.update()
