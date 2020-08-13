import unittest

import torch
from hamcrest import assert_that, equal_to

from torchvectorized.nn import EigVals
from torchvectorized.utils import sym
from torchvectorized.vlinalg import vSymEig, vExpm, vLogm, vDet


class VLinalgTest(unittest.TestCase):

    def test_should_decompose_symmetric_matrices(self):
        b, c, d, h, w = 1, 9, 32, 32, 32
        input = sym(torch.rand(b, c, d, h, w))
        eig_vals, eig_vecs = vSymEig(input, eigen_vectors=True, flatten_output=True, descending_eigen_vals=False)

        # UVU^T
        reconstructed_input = eig_vecs.bmm(torch.diag_embed(eig_vals)).bmm(eig_vecs.transpose(1, 2))
        reconstructed_input = reconstructed_input.reshape(b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w)

        assert_that(torch.allclose(reconstructed_input, input, atol=0.000001), equal_to(True))
        assert_that(torch.any(eig_vals[:, 0] > eig_vals[:, 1]), equal_to(False))
        assert_that(torch.any(eig_vals[:, 1] > eig_vals[:, 2]), equal_to(False))

        eig_vals, eig_vecs = vSymEig(input, eigen_vectors=True, flatten_output=True, descending_eigen_vals=True)

        # UVU^T
        reconstructed_input = eig_vecs.bmm(torch.diag_embed(eig_vals)).bmm(eig_vecs.transpose(1, 2))
        reconstructed_input = reconstructed_input.reshape(b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w)

        assert_that(torch.allclose(reconstructed_input, input, atol=0.000001), equal_to(True))
        assert_that(torch.any(eig_vals[:, 0] > eig_vals[:, 1]), equal_to(True))
        assert_that(torch.any(eig_vals[:, 1] > eig_vals[:, 2]), equal_to(True))

    def test_should_compute_matrices_exp_and_log(self):
        b, c, d, h, w = 1, 9, 32, 32, 32
        input = sym(torch.rand(b, c, d, h, w))
        reconstructed_input = vLogm(vExpm(input))

        assert_that(torch.allclose(reconstructed_input, input, atol=0.000001), equal_to(True))

    def test_should_decompose_identity_matrix(self):
        b, c, d, h, w = 2, 9, 1, 1, 1
        input = torch.zeros(b, c, d, h, w)
        input[:, 0, :, :, :] = 1.0
        input[:, 4, :, :, :] = 1.0
        input[:, 8, :, :, :] = 1.0

        eig_vals, eig_vecs = vSymEig(input, eigen_vectors=True, flatten_output=True)

        assert_that(torch.allclose(eig_vals, torch.ones((2, 3)), atol=0.000001), equal_to(True))
        assert_that(torch.allclose(eig_vecs, torch.eye(3).unsqueeze(0).repeat(2, 1, 1), atol=0.000001), equal_to(True))

    def test_should_decompose_matrix_with_same_diag(self):
        b, c, d, h, w = 2, 9, 1, 1, 1
        input = sym(torch.rand(b, c, d, h, w))
        input[0, 0, :, :, :] = 1.0
        input[0, 4, :, :, :] = 1.0
        input[0, 8, :, :, :] = 1.0
        input[1, :, :, :, :] = 0.0

        eig_vals, eig_vecs = vSymEig(input, eigen_vectors=True, flatten_output=True)
        expected_eig_vals, expected_eig_vecs = input.unsqueeze(0).reshape(b, 3, 3).symeig(eigenvectors=True)

        assert_that(torch.allclose(eig_vals, expected_eig_vals, atol=0.000001), equal_to(True))
        assert_that(torch.allclose(torch.abs(eig_vecs), torch.abs(expected_eig_vecs)), equal_to(True))

    def test_should_not_fail_on_empty_matrix(self):
        b, c, d, h, w = 2, 9, 1, 1, 1
        input = sym(torch.zeros(b, c, d, h, w))

        eig_vals, eig_vecs = vSymEig(input, eigen_vectors=True, flatten_output=True)

        assert_that(torch.allclose(eig_vals, torch.zeros((2, 3)), atol=0.000001), equal_to(True))
        assert_that(torch.allclose(eig_vecs, torch.eye(3).unsqueeze(0).repeat(2, 1, 1), atol=0.000001), equal_to(True))

    def test_should_compute_determinant(self):
        b, c, d, h, w = 100, 9, 1, 1, 1
        input = sym(torch.rand(b, c, d, h, w))
        det = vDet(input)
        expected_det = input.unsqueeze(0).reshape(b, 3, 3).det()

        assert_that(torch.allclose(det.flatten(), expected_det, atol=0.000001), equal_to(True))

    def test_should_compute_eigen_values(self):
        b, c, d, h, w = 1, 9, 32, 32, 32
        input = sym(torch.rand(b, c, d, h, w))
        eig_vals_expected, eig_vecs = vSymEig(input, eigen_vectors=True, flatten_output=True)

        eig_vals = EigVals()(input)

        assert_that(torch.allclose(eig_vals_expected, eig_vals, atol=0.000001), equal_to(True))


    def test_nan(self):
        inputs = torch.tensor([[[[[3.0000]]], [[[0.00]]], [[[-0.0422]]], [[[0.00]]], [[[3.0000]]], [[[-0.0194]]],
                                [[[-0.0422]]], [[[-0.0194]]], [[[3.0000]]]]])

        eig_vals, eig_vecs = vSymEig(inputs, eigen_vectors=True, flatten_output=True)
        expected_eig_vals, expected_eig_vecs = inputs.reshape(3,3).symeig(eigenvectors=True)

        reconstructed_input = eig_vecs.bmm(torch.diag_embed(eig_vals)).bmm(eig_vecs.transpose(1, 2))
        reconstructed_input = reconstructed_input.reshape(1, 1 * 1 * 1, 3, 3).permute(0, 2, 3, 1).reshape(1, 9, 1, 1, 1)
