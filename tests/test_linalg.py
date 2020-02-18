import unittest

import torch

from veigen.linalg import symeig as analytic_symeig


class EigDecompositionTest(unittest.TestCase):

    def test_should_decompose_matrix(self):
        matrix = self.sym(torch.rand(16, 9, 32, 32, 32))

        eig_vals_ana, u0, u1, u2 = analytic_symeig(matrix, eigen_vectors=True)

    def sym(self, inputs):
        return (inputs + inputs[:, [0, 3, 6, 1, 4, 7, 2, 5, 8], :, :, :]) / 2
