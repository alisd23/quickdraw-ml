import numpy as np
import os
import json

from predict import predict

## TEST EXAMPLE WITH AIRPLANE

with open('test-data/airplane_test.json') as example_data:
  example = json.load(example_data)

prediction = predict(example, {})

print(
  json.dumps(
    prediction,
    sort_keys=True,
    indent=2,
    separators=(',', ': ')
  )
)
