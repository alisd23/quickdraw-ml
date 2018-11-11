import numpy as np
import json
from tensorflow.keras.models import load_model

from src.model import top_1_accuracy, top_5_accuracy
from src.utils import load_class_names

model_filename = 'quickdraw.model.h5'
class_names = load_class_names('classes.txt')

custom_objects = {
  'top_1_accuracy': top_1_accuracy,
  'top_5_accuracy': top_5_accuracy,
}

# Load up model
model = load_model(model_filename, custom_objects)

# Expects a 28x28 array through request body
def predict(image, context):
  imageInput = np.array(image)
  imageInput = np.expand_dims(imageInput, axis=2)
  imageInput = np.expand_dims(imageInput, axis=0)

  predictions = model.predict(imageInput)[0]

  top5 = np.argsort(predictions)[-5:][::-1]
  return list(map(lambda x : { 'class': class_names[x], 'score': str(predictions[x]) }, top5))
