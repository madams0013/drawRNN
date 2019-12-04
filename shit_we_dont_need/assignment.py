
import tensorflow as tf
import numpy as np
from rnn_model import RNN_Seq2Seq
from preprocess import *

def no_noise(x):
  x = tf.dtypes.cast(x, tf.float32)
  return x

def random_noise(x):
  x = tf.dtypes.cast(x, tf.float32)
  curr = tf.random.uniform(x.shape, -0.3, 0.3)
  noised = tf.clip_by_value(x + curr, 0, 1)
  return noised

def random_scale(x):
  x = tf.dtypes.cast(x, tf.float32)
  curr = tf.random.uniform(x.shape, 0, 2.0)
  noised = tf.clip_by_value(x * curr, 0, 1)
  return noised

def call():
    