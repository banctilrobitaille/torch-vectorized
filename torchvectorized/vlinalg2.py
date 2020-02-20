from math import pi

import torch


def vSymeig2(input, eigen_vectors=False, flatten_output=False):
    b, c, d, h, w = input.size()
    input = input.double()

    eig_vals = torch.zeros(b, 3, d, h, w).to(input.device).double()
    eig_vecs = torch.zeros(b, 3, 3, d, h, w).to(input.device).double()

    nd = torch.pow(input[:, 1, :, :, :], 2) + torch.pow(input[:, 2, :, :, :], 2) + torch.pow(input[:, 5, :, :, :], 2)

    q = (input[:, 0, :, :, :] + input[:, 4, :, :, :] + input[:, 8, :, :, :]) / 3.0
    p = torch.pow((input[:, 0, :, :, :] - q), 2) + torch.pow((input[:, 4, :, :, :] - q), 2) + torch.pow(
        (input[:, 8, :, :, :] - q), 2) + 2.0 * nd
    p = torch.sqrt(p / 6.0)

    r = torch.pow((1.0 / p), 3) * ((input[:, 0, :, :, :] - q) * (
            (input[:, 4, :, :, :] - q) * (input[:, 8, :, :, :] - q) - input[:, 5, :, :, :] * input[:, 5, :, :,
                                                                                             :]) - input[:, 1, :, :,
                                                                                                   :] * (
                                           input[:, 1, :, :, :] * (input[:, 8, :, :, :] - q) - input[:, 2, :, :,
                                                                                               :] * input[:, 5, :, :,
                                                                                                    :]) + input[:, 2, :,
                                                                                                          :, :] * (
                                           input[:, 1, :, :, :] * input[:, 5, :, :, :] - input[:, 2, :, :, :] * (
                                           input[:, 4, :, :, :] - q))) / 2.0

    phi = torch.acos(r) / 3.0
    phi[r <= -1] = pi / 3
    phi[r >= 1] = 0

    eig_vals[:, 0, :, :, :] = q + 2 * p * torch.cos(phi)
    eig_vals[:, 2, :, :, :] = q + 2 * p * torch.cos(phi + pi * (2.0 / 3.0))
    eig_vals[:, 1, :, :, :] = 3 * q - eig_vals[:, 0, :, :, :] - eig_vals[:, 2, :, :, :]

    if flatten_output:
        eig_vals = eig_vals.permute(0, 2, 3, 4, 1).reshape(b * d * h * w, 3)

    return eig_vals
