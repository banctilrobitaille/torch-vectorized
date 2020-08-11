import matplotlib.pyplot as plt
import timeit
import torch

from torchvectorized.utils import sym
from torchvectorized.vlinalg import vSymEig


def vectorized_func(b):
    c, d, h, w = 9, 32, 32, 32
    input = sym(torch.rand(b, c, d, h, w))

    return timeit.timeit(lambda: vSymEig(input, eigen_vectors=False), number=5) / (5 * 1000)


def torch_func(b):
    c, d, h, w = 9, 32, 32, 32
    input = sym(torch.rand(b, c, d, h, w))
    input_flat = input.unsqueeze(1).reshape(b, 3, 3, d * h * w).permute(0, 3, 1, 2).reshape(b * d * h * w, 3, 3)
    return timeit.timeit(lambda: torch.symeig(input_flat, eigenvectors=False), number=5) / (5 * 1000)


if __name__ == "__main__":
    batch_sizes = range(0, 32)

    y1 = [vectorized_func(batch_size) for batch_size in batch_sizes]
    y2 = [torch_func(batch_size) for batch_size in batch_sizes]

    plt.plot(batch_sizes, y1, label="Torch Vectorized")
    plt.plot(batch_sizes, y2, label="Torch")
    plt.xlabel('Batch size')
    plt.ylabel('Execution time (ms)')
    plt.title('Eigendecomposition time with increasing batch size')
    plt.legend()
    plt.show()
