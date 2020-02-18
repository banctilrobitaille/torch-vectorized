import unittest

import numpy as np
import torch
from hamcrest import assert_that, equal_to

from torchvectorized.vlinalg import vSymeig


class EigDecompositionTest(unittest.TestCase):

    def test_should_decompose_matrix(self):
        b, c, d, h, w = 16, 9, 32, 32, 32
        input = self.sym(torch.rand(b, c, d, h, w))
        eig_vals, eig_vecs = vSymeig(input, eigen_vectors=True)

        eig_vals = torch.diag_embed(eig_vals.permute(0, 2, 3, 4, 1).reshape(b * d * h * w, 3))
        eig_vecs = eig_vecs.permute(0, 3, 4, 5, 1, 2).reshape(b * d * h * w, 3, 3)

        # UVU^T
        reconstructed_input = eig_vecs.bmm(eig_vals).bmm(eig_vecs.transpose(1, 2))
        reconstructed_input = reconstructed_input.reshape(b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w)

        assert_that(np.allclose(reconstructed_input.numpy(), input.numpy(), atol=0.1), equal_to(True))

    def sym(self, inputs):
        return (inputs + inputs[:, [0, 3, 6, 1, 4, 7, 2, 5, 8], :, :, :]) / 2.0
