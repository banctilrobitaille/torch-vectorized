import unittest

import torch
from PIL.GimpGradientFile import EPSILON
from hamcrest import assert_that, equal_to

from torchvectorized.vlinalg import vSymeig
from torchvectorized.vlinalg2 import vSymeig2


class EigDecompositionTest(unittest.TestCase):

    def test_should_decompose_matrix(self):
        b, c, d, h, w = 16, 9, 32, 32, 32
        input = self.sym(torch.rand(b, c, d, h, w))
        eig_vals, eig_vecs = vSymeig(input, eigen_vectors=True, flatten_output=True)

        # UVU^T
        reconstructed_input = eig_vecs.bmm(torch.diag_embed(eig_vals)).bmm(eig_vecs.transpose(1, 2))
        reconstructed_input = reconstructed_input.reshape(b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w)

        assert_that(torch.allclose(reconstructed_input, input, atol=0.000001), equal_to(True))

    def test_should_decompose_matrix_v2(self):
        b, c, d, h, w = 16, 9, 32, 32, 32
        input = -1 * self.sym(torch.ones(b, c, d, h, w))
        eig_vals, eig_vecs = vSymeig(input, eigen_vectors=True, flatten_output=True)
        eig_vals2 = vSymeig2(input, eigen_vectors=True, flatten_output=True)

        # UVU^T
        reconstructed_input = eig_vecs.bmm(torch.diag_embed(torch.exp(eig_vals))).bmm(eig_vecs.transpose(1, 2))
        reconstructed_input = reconstructed_input.reshape(b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w)

        eig_vals_1, eig_vecs = vSymeig(reconstructed_input, eigen_vectors=True, flatten_output=True)
        eig_vals2_2 = vSymeig2(reconstructed_input, eigen_vectors=True, flatten_output=True)

        assert_that(torch.allclose(reconstructed_input, input, atol=0.000001), equal_to(True))

    def sym(self, inputs):
        return (inputs + inputs[:, [0, 3, 6, 1, 4, 7, 2, 5, 8], :, :, :]) / 2.0
