import pickle
import os
from sqlitedict import SqliteDict
import zlib, sqlite3
import pandas as pd

class DataEmitter:

    def emit(self):
        raise NotImplementedError


class CachedDataEmitter(DataEmitter):

    def __init__(self, remote, path):
        self.path = path
        self.remote = remote
        self.ram_cache = None

    def emit(self):
        if self.ram_cache is not None:
            return self.ram_cache
        try:
            with open(self.path, 'rb') as file:
                self.ram_cache = pickle.load(file)
                return self.ram_cache
        except FileNotFoundError:
            self.ram_cache = self.remote.emit()
            with open(self.path, 'wb') as file:
                pickle.dump(self.ram_cache, file)
            return self.ram_cache

    def reset(self):
        self.ram_cache = None
        os.remove(self.path)


class DataEmitterGroup(DataEmitter):

    def __init__(self, keys_to_emitters_map):
        self.emitters = keys_to_emitters_map

    def emit(self):
        ret = []
        for e_key in self.emitters.keys():
            ret += [(e_key, k) for k in self.emitters[e_key].emit()]
        return ret


class KeyedDataEmitter(DataEmitter):
    def __init__(self, data_manager, keys_emitter):
        self.data_manager = data_manager
        self.keys_emitter = keys_emitter

    def emit(self):
        keys = self.keys_emitter.emit()
        values = []
        keys_ln = len(keys)
        for i, key in enumerate(keys):
            values.append(self.data_manager[key])
            if i % 1000 == 0:
                print('Emitting values ' + str(i) + '/' + str(keys_ln))
        return pd.DataFrame({'values': values}, index=keys)


def DBStorage(path):
    def db_encode(obj):
        return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

    def db_decode(obj):
        return pickle.loads(zlib.decompress(bytes(obj)))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    return SqliteDict(path, autocommit=True,
                      encode=db_encode, decode=db_decode)


class CachedDataProvider:
    def __init__(self, remote, path, converter_to_local_key=None):
        self.local = DBStorage(path)
        self.remote = remote
        self.auto_commit = True

        if converter_to_local_key is None:
            self.to_local_key = lambda key: key
        else:
            self.to_local_key = converter_to_local_key

    def __getitem__(self, key):
        local_key = self.to_local_key(key)
        try:
            return self.local[local_key]
        except KeyError:
            ret = self.remote[key]
            self.local[local_key] = ret
            return ret

    def __delitem__(self, key):
        del self.local[key]

    def force_fetch(self, key):
        try:
            del self[key]
        except KeyError:
            return self[key]

    def close(self):
        self.local.commit()
        self.local.close()


class DataManagersDispatcher:

    def __init__(self, keys_to_managers_map):
        self.managers = keys_to_managers_map

    def __getitem__(self, key):
        return self.managers[key[0]][key[1]]

    def __setitem__(self, key, value):
        self.managers[key[0]][key[1]] = value

    def __delitem__(self, key):
        del self.managers[key[0]][key[1]]