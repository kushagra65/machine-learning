import tensorflow as tf
from numpy import array
from numpy import argmax
from tf.keras.utils import to_categorical
# define example
data = [3,2,1,5,3,2,1,3,3,2,5,3,3,5]
data = array(data)
print(data)
# one hot encode
encoded = to_categorical(data)
print(encoded)
# invert encoding
inverted = argmax(encoded[0])
print(inverted)
