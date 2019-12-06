import gzip
import numpy as np

def preprocess(compressed_file, inputs_file_path):
    inputs = np.load(inputs_file_path)
    np.savez_compressed(compressed_file, a=inputs)

def compressed_to_np(compressed_file):
    loaded = np.load(compressed_file)
    print(loaded['a'])
    print(loaded['a'].shape)

preprocess('birthday_cake', 'full_numpy_bitmap_birthday cake.npy')
