import torch
from torchvectorized.utils import sym
from torchvectorized.vlinalg import vSymEig
import matplotlib.pyplot as plt

OPTIMIZER_STEPS = 5000

if __name__ == "__main__":
    cos_sim_computer = torch.nn.CosineSimilarity()

    gt_vals, gt_vecs = vSymEig(torch.rand(16, 9, 8, 8, 8), eigenvectors=True, descending_eigenvals=True)
    input = torch.nn.Parameter(sym(torch.rand(16, 9, 8, 8, 8)), requires_grad=True)

    optimizer = torch.optim.Adam([input], lr=0.001, betas=[0.5, 0.999])

    steps = []
    eig_vals_loss = []
    eig_vecs_loss = []
    cos_sim_metrics_v1 = []
    cos_sim_metrics_v2 = []
    cos_sim_metrics_v3 = []

    for optim_step in range(OPTIMIZER_STEPS):
        optimizer.zero_grad()
        eig_vals, eig_vecs = vSymEig(input, eigenvectors=True, descending_eigenvals=True)
        loss_eig_val = torch.nn.functional.l1_loss(eig_vals, gt_vals)
        loss_eig_vecs = torch.nn.functional.l1_loss(eig_vecs, gt_vecs)

        total_loss = loss_eig_vecs + loss_eig_val

        steps.append(optim_step)
        eig_vals_loss.append(loss_eig_val)
        eig_vecs_loss.append(loss_eig_vecs)
        cos_sim_metrics_v1.append(
            torch.abs(cos_sim_computer(eig_vecs[:, :, 0, :, :, :], gt_vecs[:, :, 0, :, :, :])).mean())
        cos_sim_metrics_v2.append(
            torch.abs(cos_sim_computer(eig_vecs[:, :, 1, :, :, :], gt_vecs[:, :, 1, :, :, :])).mean())
        cos_sim_metrics_v3.append(
            torch.abs(cos_sim_computer(eig_vecs[:, :, 2, :, :, :], gt_vecs[:, :, 2, :, :, :])).mean())

        total_loss.backward()
        optimizer.step()

    plt.plot(steps, eig_vals_loss, label="Eigenvalues L1 error")
    plt.plot(steps, eig_vecs_loss, label="Eigenvectors L1 error")
    plt.plot(steps, cos_sim_metrics_v1, label="Cosine Similarity \u03B51")
    plt.plot(steps, cos_sim_metrics_v2, label="Cosine Similarity \u03B52")
    plt.plot(steps, cos_sim_metrics_v3, label="Cosine Similarity \u03B53")
    plt.xlabel('Step')
    plt.legend()
    plt.show()
