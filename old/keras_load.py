from tensorflow import keras
import sys

model = keras.models.load_model(sys.argv[1])

