import tensorflow as tf

print(tf.__version__)

if tf.__version__ != '2.13':
    print('Please install TensorFlow version 2.13')
else:
    print('TensorFlow version is correct')