import unittest

from life import grid_adjacency
from life import count_alive_neighbors_matmul
from life import count_alive_neighbors_slice
import numpy as np


def neighbors(i, j, m, n):
    inbrs = [-1, 0, 1]
    if i == 0:
        inbrs = [0, 1]
    if i == m-1:
        inbrs = [-1, 0]
    jnbrs = [-1, 0, 1]
    if j == 0:
        jnbrs = [0, 1]
    if j == n-1:
        jnbrs = [-1, 0]

    for delta_i in inbrs:
        for delta_j in jnbrs:
            if delta_i == delta_j == 0:
                continue
            yield i + delta_i, j + delta_j

def count_alive_neighbors(S):
    m, n = S.shape
    cts = np.zeros(S.shape, dtype=np.int64)
    for i in range(m):
        for j in range(n):
            for i2, j2 in neighbors(i, j, m, n):
                cts[i,j] = cts[i,j] + S[i2, j2]

    return cts



class TestAdjacency(unittest.TestCase):

	def setUp(self):
		pass

	def test_grid_adjacency(self):

		Atrue = np.array([[0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
			[1., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0.],
			[0., 1., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0.],
			[0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
			[1., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0.],
			[1., 1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0.],
			[0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 1., 1.],
			[0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 1.],
			[0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0.],
			[0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0.],
			[0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1.],
			[0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0.]])

		self.assertTrue(np.all(grid_adjacency(3,4).toarray() == Atrue))

		Atrue = np.array([[0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
			[1., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
			[0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
			[1., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0.],
			[1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0.],
			[0., 1., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0.],
			[0., 0., 0., 1., 1., 0., 0., 1., 0., 1., 1., 0.],
			[0., 0., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
			[0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 1., 1.],
			[0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0.],
			[0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 1.],
			[0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0.]])

		self.assertTrue(np.all(grid_adjacency(4,3).toarray() == Atrue))


class TestMatmul(unittest.TestCase):

	def setUp(self):
		pass

	def test_slice(self):
		m = 10
		n = 20
		for s in [0,1,2]:
			np.random.seed(s)
			S = np.random.rand(m+s, n+s) < 0.3
			A = grid_adjacency(m+s, n+s)

			Ctrue = count_alive_neighbors(S)
			Cmm = count_alive_neighbors_matmul(S, A)

			self.assertTrue(np.all(Ctrue == Cmm))


class TestSlice(unittest.TestCase):

	def setUp(self):
		pass

	def test_slice(self):
		m = 10
		n = 20
		for s in [0,1,2]:
			np.random.seed(s)
			S = np.random.rand(m+s, n+s) < 0.3

			Ctrue = count_alive_neighbors(S)
			Cslice = count_alive_neighbors_slice(S)

			self.assertTrue(np.all(Ctrue == Cslice))
