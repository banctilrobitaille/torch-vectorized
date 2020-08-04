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

        return torch.clamp(torch.sqrt(num / (denom + 0.0000001)), 0, 1)

    def test_should_backward_eig_vals(self):
        grad_X = torch.ones(1, 3)

        spd = torch.empty(1, 9, 1, 1, 1)
        spd[0, :, 0, 0, 0] = torch.Tensor([4.2051, 1.1989, 0.6229, 1.1989, 4.1973, 0.6028, 0.6229, 0.6028, 3.5204])

        S, U = vSymEig(spd, eigen_vectors=True, flatten_output=True)

        fa = self.compute_fa(S)
        gradients = self.backward_eig_vals(S, U, spd, grad_X)

    def backward_eig_vals(self, S, U, X, grad_X):
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        b, c, d, h, w = X.size()
        grad_X = torch.diag_embed(grad_X)
        grad_X = grad_X.reshape(b, 3, 3, d * h * w).permute(0, 3, 1, 2).reshape(b * d * h * w, 3, 3)

        grad_U = 2 * self.sym_grad(grad_X).bmm(U.bmm(torch.diag_embed(S)))
        grad_S = torch.eye(3).to(grad_X.device) * torch.diag_embed(S).bmm(
            U.transpose(1, 2).bmm(self.sym_grad(grad_X).bmm(U)))

        S = S.view(1, -1)
        P = S.view(S.size(1) // 3, 3).unsqueeze(2)
        P = P.expand(P.size(0), P.size(1), 3)
        P = P - P.transpose(1, 2)
        mask_zero = torch.abs(P) == 0
        P = 1 / P
        P[mask_zero] = 0

        return U.bmm(self.sym_grad(P.transpose(1, 2) * (U.transpose(1, 2).bmm(grad_U))) + grad_S).bmm(
            U.transpose(1, 2)).reshape(
            b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w), None

    def test_should_compute_eigen_values(self):
        b, c, d, h, w = 1, 9, 32, 32, 32
        input = self.sym(torch.rand(b, c, d, h, w))
        eig_vals_expected, eig_vecs = vSymEig(input, eigen_vectors=True)

        eig_vals = EigVals()(input)

        assert_that(torch.allclose(eig_vals_expected, eig_vals, atol=0.000001), equal_to(True))

    def sym(self, inputs):
        return (inputs + inputs[:, [0, 3, 6, 1, 4, 7, 2, 5, 8], :, :, :]) / 2.0

    def sym_grad(self, X):
        return 0.5 * (X + X.transpose(1, 2))
