from typing import Dict, List
from collections import OrderedDict
import numpy as np


def euclidean_dist(x: List, y: List) -> float:
    return np.linalg.norm(np.array(x) - np.array(y))


class LSH:
    def __init__(self, hash_byte_len: int, num_hashes: int, num_tables: int):
        self.hash_byte_len = hash_byte_len
        self.num_hashes = num_hashes
        self.num_tables = num_tables

        self.planes = [
            np.random.randn(self.hash_byte_len, self.num_hashes) for _ in range(self.num_tables)
        ]

        self.hash_tables: List[Dict] = [dict() for i in range(self.num_tables)]

    def hash(self, plane: np.ndarray, vector: List) -> str:
        projections = np.dot(plane, np.array(vector))
        return "".join(["1" if i > 0 else "0" for i in projections])

    def index(self, vector: List, data: str) -> None:
        for loc, htable in enumerate(self.hash_tables):
            this_hash = self.hash(self.planes[loc], vector)
            htable.setdefault(this_hash, []).append((vector, data))

    def query(self, vector: List, k: int = None, dist_func=euclidean_dist):
        # [y for x in a for y in [x[0]] * x[1]]
        contestants = []
        for loc, htable in enumerate(self.hash_tables):
            this_hash = self.hash(self.planes[loc], vector)
            contestants.extend(htable.get(this_hash))
        candidates = [(contestent, dist_func(vector, contestent[0])) for contestent in contestants]
        candidates.sort(key=lambda x: x[1])
        candidate_data = list(OrderedDict.fromkeys([candidate[0][1] for candidate in candidates]))
        return candidate_data[:k] if k else candidate_data
