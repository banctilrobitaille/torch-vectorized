import unittest

import torch
from hamcrest import assert_that, equal_to

from torchvectorized.nn.spd_veig import EigVals
from torchvectorized.vlinalg import vSymEig


class SpdVEigTest(unittest.TestCase):

    def compute_fa(self, eigen_values: torch.Tensor):
        eig_1 = eigen_values[:, 0]
        eig_2 = eigen_values[:, 1]
        eig_3 = eigen_values[:, 2]

        num = (eig_1 - eig_2) ** 2 + (eig_2 - eig_3) ** 2 + (eig_1 - eig_3) ** 2
        denom = 2 * (eig_1 ** 2 + eig_2 ** 2 + eig_3 ** 2)

        return torch.clamp(torch.sqrt((num / (denom + 1e-15) + 1e-15)), 0, 1)

    def test_should_backward_eig_vals(self):
        b, c, d, h, w = 1, 9, 32, 32, 32
        real = self.sym(torch.rand(b, c, d, h, w))
        fake = self.sym(torch.rand(b, c, d, h, w))
        real = torch.eye(3).mm(torch.diag(torch.tensor([0.0, 0.0, 0.0]))).mm(torch.eye(3).T).reshape(1, 9, 1, 1, 1)
        fake = torch.eye(3).mm(torch.diag(torch.tensor([0.0, 0.0, 0.0]))).mm(torch.eye(3).T).reshape(1, 9, 1, 1, 1)

        real.requires_grad = False
        fake.requires_grad = True

        loss_fn = torch.nn.L1Loss()
        to_eig_vals = EigVals()

        real = torch.tensor([-7.0249, -0.0672, 0.0503, -0.0672, -6.8831, 0.0353, 0.0503, 0.0353,
                             -7.1394]).reshape(1, 9, 1, 1, 1)
        real_1 = torch.tensor([0.9781, -0.0028, 0.0091, -0.0028, 1.0559, -0.0112, 0.0091, -0.0112, 0.8748],
                              requires_grad=True).reshape(1, 9, 1, 1, 1)

        real_2 = torch.tensor([1.0567, 0.0000, 0.0000, 0.0000, 0.9787, 0.0000, 0.0000, 0.0000, 0.8734],
                              requires_grad=True).reshape(1, 9, 1, 1, 1)

        eig_1 = to_eig_vals(fake)
        eig_2 = to_eig_vals(real)
        eig_3 = to_eig_vals(real_1)
        eig_4 = to_eig_vals(real_2)

        fa_1 = self.compute_fa(torch.exp(eig_1))
        fa_2 = self.compute_fa(torch.exp(eig_2))
        fa_3 = self.compute_fa(torch.exp(eig_3))
        fa_4 = self.compute_fa(torch.exp(eig_4))

        loss = loss_fn(fa_3, fa_4)
        loss.backward()

        print("Hello")

    def test_should_compute_eigen_values(self):
        b, c, d, h, w = 1, 9, 32, 32, 32
        input = self.sym(torch.rand(b, c, d, h, w))
        eig_vals_expected, eig_vecs = vSymEig(input, eigen_vectors=True, flatten_output=True)

        eig_vals = EigVals()(input)

        assert_that(torch.allclose(eig_vals_expected, eig_vals, atol=0.000001), equal_to(True))

    def sym(self, inputs):
        return (inputs + inputs[:, [0, 3, 6, 1, 4, 7, 2, 5, 8], :, :, :]) / 2.0

    def sym_grad(self, X):
        return 0.5 * (X + X.transpose(1, 2))
