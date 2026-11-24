import pickle

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

