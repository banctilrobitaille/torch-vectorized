import unittest

import torch
from hamcrest import assert_that, equal_to

from torchvectorized.debug.nn import Logm, EigVals, Expm
from torchvectorized.utils import sym
from torchvectorized.vlinalg import vSymEig


class NnTest(unittest.TestCase):

    def setUp(self):
        self._target = torch.tensor(
            [6.7756e-04, 3.4127e-05, -1.1893e-04, 3.4127e-05, 7.4100e-04, 9.1343e-05, -1.1893e-04, 9.1343e-05,
             .7955e-04]).reshape(1, 9, 1, 1, 1)
        self._target_log = torch.tensor(
            [-7.31479692, 0.06039926, -0.18166597, 0.06039926, -7.21763706, 0.13518471, -0.18166597, 0.13518471,
             -7.31935453]).reshape(1, 9, 1, 1, 1)
        self._pred = torch.tensor(
            [6.3381e-04, -1.0740e-04, -1.1193e-05, -1.0740e-04, 5.1552e-04, 2.0420e-05, -1.1193e-05, 2.0420e-05,
             2.8239e-04]).reshape(1, 9, 1, 1, 1)
        self._pred_log = torch.tensor(
            [-7.38063335, - 0.1894127, - 0.02080916, - 0.1894127, - 7.59057522, 0.05062385, - 0.02080916, 0.05062385,
             - 8.17417812]).reshape(1, 9, 1, 1, 1)

    def compute_fa(self, eigen_values: torch.Tensor):
        eig_1 = eigen_values[:, 0]
        eig_2 = eigen_values[:, 1]
        eig_3 = eigen_values[:, 2]

        num = (eig_1 - eig_2) ** 2 + (eig_2 - eig_3) ** 2 + (eig_1 - eig_3) ** 2
        denom = 2 * (eig_1 ** 2 + eig_2 ** 2 + eig_3 ** 2)

        return torch.clamp(torch.sqrt((num / (denom + 1e-15) + 1e-15)), 0, 1)

    def test_should_backward_eig_vals(self):
        self._target_log.requires_grad = False
        self._pred_log.requires_grad = True

        loss_fn = torch.nn.L1Loss()
        to_eig_vals = EigVals()

        eig_1 = to_eig_vals(self._pred_log)
        eig_2 = to_eig_vals(self._target_log)

        eig_1_torch, vec_1 = self._pred_log.reshape(1, 3, 3).symeig(eigenvectors=True)
        eig_2_torch, vec_2 = self._target_log.reshape(1, 3, 3).symeig(eigenvectors=True)

        fa_1 = self.compute_fa(torch.exp(eig_1))
        fa_2 = self.compute_fa(torch.exp(eig_2))

        fa_1_torch = self.compute_fa(torch.exp(eig_1_torch))
        fa_2_torch = self.compute_fa(torch.exp(eig_2_torch))

        assert_that(torch.allclose(fa_1_torch, fa_1, atol=0.000001), equal_to(True))
        assert_that(torch.allclose(fa_2_torch, fa_2, atol=0.000001), equal_to(True))

        loss = loss_fn(fa_1, fa_2)
        loss_torch = loss_fn(fa_1_torch, fa_2_torch)

        loss.backward()
        grad = self._pred_log.grad.clone()
        self._pred_log.grad = None
        loss_torch.backward()
        grad_torch = self._pred_log.grad

        assert_that(torch.allclose(grad, grad_torch, atol=0.000001), equal_to(True))

    def test_should_backward_logm(self):
        self._target_log.requires_grad = False
        self._pred.requires_grad = True

        loss_fn = torch.nn.L1Loss()
        logm = Logm()

        pred_log = logm(self._pred)
        S, U = self._pred.reshape(3, 3).symeig(eigenvectors=True)

        pred_log_torch = U.reshape(1, 3, 3).bmm(torch.diag_embed(torch.log(S.reshape(1, 3)))).bmm(
            U.reshape(1, 3, 3).transpose(1, 2)).reshape(1, 9, 1, 1, 1)

        assert_that(torch.allclose(pred_log_torch, self._pred_log, atol=0.00001), equal_to(True))
        assert_that(torch.allclose(pred_log, self._pred_log, atol=0.00001), equal_to(True))

        loss = loss_fn(pred_log, self._target_log)
        loss_torch = loss_fn(pred_log_torch, self._target_log)

        loss.backward()
        grad = self._pred.grad.clone()
        self._pred.grad = None
        loss_torch.backward()
        grad_torch = self._pred.grad

        assert_that(torch.allclose(grad, grad_torch, atol=0.000001), equal_to(True))

    def test_should_backward_expm(self):
        self._target.requires_grad = False
        self._pred_log.requires_grad = True

        loss_fn = torch.nn.L1Loss()
        expm = Expm()

        pred = expm(self._pred_log)
        S, U = self._pred_log.reshape(3, 3).symeig(eigenvectors=True)

        pred_torch = U.reshape(1, 3, 3).bmm(torch.diag_embed(torch.exp(S.reshape(1, 3)))).bmm(
            U.reshape(1, 3, 3).transpose(1, 2)).reshape(1, 9, 1, 1, 1)

        assert_that(torch.allclose(pred_torch, self._pred, atol=0.00001), equal_to(True))
        assert_that(torch.allclose(pred, self._pred, atol=0.00001), equal_to(True))

        loss = loss_fn(pred, self._target)
        loss_torch = loss_fn(pred_torch, self._target)

        loss.backward()
        grad = self._pred_log.grad.clone()
        self._pred_log.grad = None
        loss_torch.backward()
        grad_torch = self._pred_log.grad

        assert_that(torch.allclose(grad, grad_torch, atol=0.000001), equal_to(True))

    def test_should_compute_eigen_values(self):
        b, c, d, h, w = 1, 9, 32, 32, 32
        input = sym(torch.rand(b, c, d, h, w))
        eig_vals_expected, eig_vecs = vSymEig(input, eigen_vectors=True, flatten_output=True)

        eig_vals = EigVals()(input)

        assert_that(torch.allclose(eig_vals_expected, eig_vals, atol=0.000001), equal_to(True))
