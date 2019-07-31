from typing import Dict, List
import numpy as np

# The idea here will be to replace the hash_tables with elasticsearch.
def euclidean_dist(x, y):
    diff = np.array(x) - y
    return np.sqrt(np.dot(diff, diff))

class LSH:
    def __init__(self, hash_byte_len: int, dims: int, tables: int):
        self.hash_byte_len = hash_byte_len
        self.dims = dims
        self.num_tables = tables

        # for unit tests, manually set planes to known values.
        self.planes = [np.random.randn(self.hash_byte_len, self.dims)
                       for _ in range(self.num_tables)]

        print(self.planes)
        self.hash_tables: List[Dict] = [dict() for i in range(self.num_tables)]

    def hash(self, planes, point):
        try:
            projections = np.dot(planes, np.array(point))
        except TypeError as e:
            print(e)
            raise
        except ValueError as e:
            print(e)
            raise
        return "".join(["1" if i > 0 else "0" for i in projections])

    def index(self, point, data=None):
        if isinstance(point, np.ndarray):
            point = point.tolist()
        if data:
            value = {tuple(point), data}
        else:
            value = tuple(point)

        print(f"VAL: {value}")
        for loc, htable in enumerate(self.hash_tables):
            this_hash = self.hash(self.planes[loc], point)
            htable.setdefault(this_hash, []).append(value)

    def query(self, point, k=None, dist_func=euclidean_dist):
        contestants = set()
        for loc, htable in enumerate(self.hash_tables):
            b_hash = self.hash(self.planes[loc], point)
            contestants.update(htable.get(b_hash))
        candidates = [(minhash, dist_func(point, self._as_array(minhash))) for minhash in contestants]
        candidates.sort(key=lambda x: x[1])
        return candidates[:k] if k else candidates

    def _as_array(self, keys):
        if isinstance(keys[0], tuple):
            return np.asarray(keys[0])
        # elif isinstance(keys, (tuple, list)):
        try:
            return np.asarray(keys)
        except ValueError as err:
            print(err)
            raise
        else:
            raise TypeError


