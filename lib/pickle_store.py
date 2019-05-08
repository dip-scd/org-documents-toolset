import os
import pickle

class PickleStore:

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(self.path),
                    exist_ok=True)

    def set(self, value):
        with open(self.path, 'wb') as file:
            pickle.dump(value, file)

    def get(self):
        try:
            with open(self.path, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None