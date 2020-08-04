from math import pi

import torch


def _get_diag_matrices_index(input: torch.Tensor):
    a12 = input[:, 1, :, :, :].double()
    a13 = input[:, 2, :, :, :].double()
    a23 = input[:, 5, :, :, :].double()
    nd = torch.pow(a12, 2) + torch.pow(a13, 2) + torch.pow(a23, 2)

    return torch.where(nd == 0)


def _compute_eigen_values(input: torch.Tensor):
    b, c, d, h, w = input.size()
    a11 = input[:, 0, :, :, :].double()
    a12 = input[:, 1, :, :, :].double()
    a13 = input[:, 2, :, :, :].double()
    a22 = input[:, 4, :, :, :].double()
    a23 = input[:, 5, :, :, :].double()
    a33 = input[:, 8, :, :, :].double()
    eig_vals = torch.zeros(b, 3, d, h, w).to(input.device).double()

    nd = torch.pow(a12, 2) + torch.pow(a13, 2) + torch.pow(a23, 2)

    if torch.any(nd != 0):
        q = (a11 + a22 + a33) / 3.0
        p = torch.pow((a11 - q), 2) + torch.pow((a22 - q), 2) + torch.pow((a33 - q), 2) + 2.0 * nd
        p = torch.sqrt(p / 6.0)

        r = torch.pow((1.0 / p), 3) * ((a11 - q) * ((a22 - q) * (a33 - q) - a23 * a23) - a12 * (
                a12 * (a33 - q) - a13 * a23) + a13 * (a12 * a23 - a13 * (a22 - q))) / 2.0

        phi = torch.acos(r) / 3.0
        phi[r <= -1] = pi / 3
        phi[r >= 1] = 0

        eig_vals[:, 0, :, :, :] = q + 2 * p * torch.cos(phi)
        eig_vals[:, 2, :, :, :] = q + 2 * p * torch.cos(phi + pi * (2.0 / 3.0))
        eig_vals[:, 1, :, :, :] = 3 * q - eig_vals[:, 0, :, :, :] - eig_vals[:, 2, :, :, :]

    if torch.any(nd == 0):
        diag_matrix_index = torch.where(nd == 0)
        eig_vals[:, 0, :, :, :][diag_matrix_index] = a11[diag_matrix_index]
        eig_vals[:, 1, :, :, :][diag_matrix_index] = a22[diag_matrix_index]
        eig_vals[:, 2, :, :, :][diag_matrix_index] = a33[diag_matrix_index]

    return eig_vals


def _compute_eigen_vectors(input: torch.Tensor, eigen_values: torch.Tensor):
    nd = input[:, 1, :, :, :].unsqueeze(1).double() * \
         input[:, 2, :, :, :].unsqueeze(1).double() * \
         input[:, 5, :, :, :].unsqueeze(1).double()
    a11 = input[:, 0, :, :, :].unsqueeze(1).expand(eigen_values.size()).double()
    a12 = input[:, 1, :, :, :].unsqueeze(1).expand(eigen_values.size()).double()
    a13 = input[:, 2, :, :, :].unsqueeze(1).expand(eigen_values.size()).double()
    a22 = input[:, 4, :, :, :].unsqueeze(1).expand(eigen_values.size()).double()
    a23 = input[:, 5, :, :, :].unsqueeze(1).expand(eigen_values.size()).double()

    u0 = a12 * a23 - a13 * (a22 - eigen_values)
    u1 = a12 * a13 - a23 * (a11 - eigen_values)
    u2 = (a11 - eigen_values) * (a22 - eigen_values) - a12 * a12
    norm = torch.sqrt(torch.pow(u0, 2) + torch.pow(u1, 2) + torch.pow(u2, 2))
    u0 = u0 / norm
    u1 = u1 / norm
    u2 = u2 / norm

    if torch.any(((eigen_values[:, 0, :, :, :] == eigen_values[:, 1, :, :, :]).float() * (
            eigen_values[:, 0, :, :, :] == eigen_values[:, 2, :, :, :]).float()) == 1.0):
        index = torch.where(
            (eigen_values[:, 0, :, :, :] + eigen_values[:, 1, :, :, :] + eigen_values[:, 2, :, :, :]) == (
                    3 * eigen_values[:, 0, :, :, :]))
        u0[index[0], :, index[1], index[2], index[3]] = torch.tensor([1, 0, 0]).to(input.device).double()
        u1[index[0], :, index[1], index[2], index[3]] = torch.tensor([0, 1, 0]).to(input.device).double()
        u2[index[0], :, index[1], index[2], index[3]] = torch.tensor([0, 0, 1]).to(input.device).double()

    if torch.any(nd == 0):
        index = torch.where(nd == 0)
        u0[index[0], :, index[1], index[2], index[3]] = torch.tensor([1, 0, 0]).to(input.device).double()
        u1[index[0], :, index[1], index[2], index[3]] = torch.tensor([0, 1, 0]).to(input.device).double()
        u2[index[0], :, index[1], index[2], index[3]] = torch.tensor([0, 0, 1]).to(input.device).double()

    return torch.cat([u0.unsqueeze(1), u1.unsqueeze(1), u2.unsqueeze(1)], dim=1)


def vSymEig(input: torch.Tensor, eigen_vectors=False, flatten_output=False):
    eig_vals = _compute_eigen_values(input)

    if eigen_vectors:
        eig_vecs = _compute_eigen_vectors(input, eig_vals)
    else:
        eig_vecs = None

    if flatten_output:
        b, c, d, h, w = input.size()
        eig_vals = eig_vals.permute(0, 2, 3, 4, 1).reshape(b * d * h * w, 3)
        eig_vecs = eig_vecs.permute(0, 3, 4, 5, 1, 2).reshape(b * d * h * w, 3, 3) if eigen_vectors else eig_vecs

    return eig_vals.float(), eig_vecs.float() if eig_vecs is not None else None


def vExpm(input: torch.Tensor, replace_nans=False):
    b, c, d, h, w = input.size()
    eig_vals, eig_vecs = vSymEig(input, eigen_vectors=True, flatten_output=True)

    # UVU^T
    reconstructed_input = eig_vecs.bmm(torch.diag_embed(torch.exp(eig_vals))).bmm(eig_vecs.transpose(1, 2))
    output = reconstructed_input.reshape(b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w)

    if replace_nans:
        output[torch.where(torch.isnan(output))] = 0

    return output


def vLogm(input: torch.Tensor, replace_nans=False):
    b, c, d, h, w = input.size()
    eig_vals, eig_vecs = vSymEig(input, eigen_vectors=True, flatten_output=True)

    # UVU^T
    reconstructed_input = eig_vecs.bmm(torch.diag_embed(torch.log(eig_vals))).bmm(eig_vecs.transpose(1, 2))
    output = reconstructed_input.reshape(b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w)

    if replace_nans:
        output[torch.where(torch.isnan(output))] = 0

    return output


def vTrace(input: torch.Tensor):
    return input[:, 0, :, :, :] + input[:, 4, :, :, :] + input[:, 8, :, :, :]


def vDet(input: torch.Tensor):
    a = input[:, 0, :, :, :].double()
    b = input[:, 1, :, :, :].double()
    c = input[:, 2, :, :, :].double()
    d = input[:, 4, :, :, :].double()
    e = input[:, 5, :, :, :].double()
    f = input[:, 8, :, :, :].double()
    return (a * (d * f - (e ** 2)) + b * (c * e - (b * f)) + c * (b * e - (d * c))).float()
