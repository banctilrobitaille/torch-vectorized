import unittest

import torch
from hamcrest import assert_that, equal_to

from torchvectorized.nn import EigVals
from torchvectorized.utils import sym
from torchvectorized.vlinalg import vSymEig


class NnTest(unittest.TestCase):

    def compute_fa(self, eigen_values: torch.Tensor):
        eig_1 = eigen_values[:, 0]
        eig_2 = eigen_values[:, 1]
        eig_3 = eigen_values[:, 2]

        num = (eig_1 - eig_2) ** 2 + (eig_2 - eig_3) ** 2 + (eig_1 - eig_3) ** 2
        denom = 2 * (eig_1 ** 2 + eig_2 ** 2 + eig_3 ** 2)

        return torch.clamp(torch.sqrt((num / (denom + 1e-15) + 1e-15)), 0, 1)

    def test_should_backward_eig_vals(self):
        real_log = torch.tensor(
            [-7.31479692, 0.06039926, - 0.18166597, 0.06039926, - 7.21763706, 0.13518471, - 0.18166597, 0.13518471,
             - 7.31935453]).reshape(1, 9, 1, 1, 1)
        fake_log = torch.tensor(
            [-7.38063335, - 0.1894127, - 0.02080916, - 0.1894127, - 7.59057522, 0.05062385, - 0.02080916, 0.05062385,
             - 8.17417812]).reshape(1, 9, 1, 1, 1)

        real_log.requires_grad = False
        fake_log.requires_grad = True

        loss_fn = torch.nn.L1Loss()
        to_eig_vals = EigVals()

        eig_1 = to_eig_vals(fake_log)
        eig_2 = to_eig_vals(real_log)

        eig_1_torch, vec_1 = fake_log.reshape(1, 3, 3).symeig(eigenvectors=True)
        eig_2_torch, vec_2 = real_log.reshape(1, 3, 3).symeig(eigenvectors=True)

        fa_1 = self.compute_fa(torch.exp(eig_1))
        fa_2 = self.compute_fa(torch.exp(eig_2))

        fa_1_torch = self.compute_fa(torch.exp(eig_1_torch))
        fa_2_torch = self.compute_fa(torch.exp(eig_2_torch))

        assert_that(torch.allclose(fa_1_torch, fa_1, atol=0.000001), equal_to(True))
        assert_that(torch.allclose(fa_2_torch, fa_2, atol=0.000001), equal_to(True))

        loss = loss_fn(fa_1, fa_2)
        loss_torch = loss_fn(fa_1_torch, fa_2_torch)

        loss.backward()
        grad = fake_log.grad
        loss_torch.backward()
        grad_torch = fake_log.grad

        assert_that(torch.allclose(grad, grad_torch, atol=0.000001), equal_to(True))

    def test_should_compute_eigen_values(self):
        b, c, d, h, w = 1, 9, 32, 32, 32
        input = sym(torch.rand(b, c, d, h, w))
        eig_vals_expected, eig_vecs = vSymEig(input, eigen_vectors=True, flatten_output=True)

        eig_vals = EigVals()(input)

        assert_that(torch.allclose(eig_vals_expected, eig_vals, atol=0.000001), equal_to(True))
