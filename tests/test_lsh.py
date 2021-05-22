from unittest import TestCase
import numpy as np
from lsh import LSH


class TestLsh(TestCase):
    """TODO: Test Case docstring goes here."""

    def setUp(self):
        self.lsh = LSH(3, 2, 1)
        self.lsh_two_tables = LSH(3, 2, 2)

        # Overwrite randomly initalized planes with known values.
        self.lsh.planes = [np.array([[0.1, 0.2], [-0.1, -0.2], [-1.0, 1.0]])]
        self.lsh_two_tables.planes = [
            np.array([[0.1, 0.2], [-0.1, -0.2], [-1.0, 1.0]]),
            np.array([[-0.1, -0.2], [0.1, 0.2], [-2.0, 2.0]]),
        ]

    def test_hashing(self):
        vector_ones = [1, 1]
        # This will add each plane without a scalar.
        # each value greater than zero will append a 1 to the string, 0 otherwise.
        self.assertEqual(self.lsh.hash(self.lsh.planes[0], vector_ones), "100")

        vector_twos = [-2, 2]
        self.assertEqual(self.lsh.hash(self.lsh.planes[0], vector_twos), "101")

    def test_table_indexing(self):
        self.lsh.index([1, 1], "data1")
        self.lsh.index([-2, 2], "data2")
        self.assertDictEqual(
            self.lsh.hash_tables[0], {"100": [([1, 1], "data1")], "101": [([-2, 2], "data2")]}
        )

        self.lsh_two_tables.index([1, 1], "data1")
        self.lsh_two_tables.index([-2, 2], "data2")
        self.assertDictEqual(
            self.lsh_two_tables.hash_tables[0],
            {"100": [([1, 1], "data1")], "101": [([-2, 2], "data2")]},
        )
        self.assertDictEqual(
            self.lsh_two_tables.hash_tables[1],
            {"010": [([1, 1], "data1")], "011": [([-2, 2], "data2")]},
        )

    def test_query(self):
        self.lsh.index([1, 1], "data1")
        self.lsh.index([-2, 2], "data2")
        output = self.lsh.query([1, 1], 1)
        self.assertEqual(output, ["data1"])

        self.lsh_two_tables.index([1, 1], "data1")
        self.lsh_two_tables.index([-2, 2], "data2")
        output = self.lsh_two_tables.query([1, 1], 1)
        self.assertEqual(output, ["data1"])

        self.lsh_two_tables.index([-1, -1], "data3")
        self.lsh_two_tables.index([6, 6], "data4")
        self.lsh_two_tables.index([-10, -10], "data5")
        output = self.lsh_two_tables.query([6, 6], 2)
        self.assertEqual(output, ["data4", "data1"])
