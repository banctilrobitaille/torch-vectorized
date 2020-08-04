# import pydevd
import torch
from torchvectorized.vlinalg import vSymEig


def sym(X):
    return 0.5 * (X + X.transpose(1, 2))


class ToEigVals(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        V, U = vSymEig(X, eigen_vectors=True, flatten_output=False)
        ctx.save_for_backward(V, U, X)

        return V

    @staticmethod
    def backward(ctx, *grad_outputs):
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        S, U, X = ctx.saved_tensors
        b, c, d, h, w = X.size()
        S = S.permute(0, 2, 3, 4, 1).reshape(b * d * h * w, 3)

        grad_X = torch.diag_embed(grad_outputs[0].permute(0, 2, 3, 4, 1).reshape(b * d * h * w, 3))
        grad_X = grad_X.reshape(b, 3, 3, d * h * w).permute(0, 3, 1, 2).reshape(b * d * h * w, 3, 3)

        grad_U = 2 * sym(grad_X).bmm(U.bmm(torch.diag_embed(S)))
        grad_S = torch.eye(3).to(grad_X.device) * torch.diag_embed(S).bmm(U.transpose(1, 2).bmm(sym(grad_X).bmm(U)))

        S = S.view(1, -1)
        P = S.view(S.size(1) // 3, 3).unsqueeze(2)
        P = P.expand(P.size(0), P.size(1), 3)
        P = P - P.transpose(1, 2)
        mask_zero = torch.abs(P) == 0
        P = 1 / P
        P[mask_zero] = 0

        return U.bmm(sym(P.transpose(1, 2) * (U.transpose(1, 2).bmm(grad_U))) + grad_S).bmm(U.transpose(1, 2)).reshape(
            b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w), None


class EigVals(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return ToEigVals.apply(X)
