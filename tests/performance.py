import matplotlib.pyplot as plt
import timeit
import torch

from torchvectorized.utils import sym
from torchvectorized.vlinalg import vSymEig


def rand_torch_input(shape):
    b, c, d, h, w = shape
    input = sym(torch.rand(b, c, d, h, w))
    return input.unsqueeze(1).reshape(b, 3, 3, d * h * w).permute(0, 3, 1, 2).reshape(b * d * h * w, 3, 3)


def vectorized_func(b):
    c, d, h, w = 9, 10, 10, 10
    return timeit.timeit(lambda: vSymEig(sym(torch.rand(b, c, d, h, w)), eigenvectors=True), number=5) / (5 * 1000000)


def torch_func(b):
    c, d, h, w = 9, 10, 10, 10
    return timeit.timeit(lambda: torch.symeig(rand_torch_input((b, c, d, h, w)), eigenvectors=True), number=5) / 5


def vectorized_func_gpu(b):
    c, d, h, w = 9, 10, 10, 10
    return timeit.timeit(lambda: vSymEig(sym(torch.rand(b, c, d, h, w)).cuda(), eigenvectors=True), number=5) / 5


def torch_func_gpu(b):
    c, d, h, w = 9, 10, 10, 10
    return timeit.timeit(lambda: torch.symeig(rand_torch_input((b, c, d, h, w)).cuda(), eigenvectors=True),
                         number=5) / 5


if __name__ == "__main__":
    batch_sizes = range(1, 500)

    """y1 = [vectorized_func(batch_size) for batch_size in batch_sizes]
    y2 = [torch_func(batch_size) for batch_size in batch_sizes]

    plt.plot([batch_size * 1000 for batch_size in batch_sizes], y1, label="Torch Vectorized")
    plt.plot([batch_size * 1000 for batch_size in batch_sizes], y2, label="Torch")
    plt.xlabel('Batch size')
    plt.ylabel('Execution time (\u03BCs)')
    plt.title('SymEig exec. time (\u03BCs) vs batch size (K) (CPU)')
    plt.legend()
    plt.savefig("cpu_performance.png")
    plt.close()

    if torch.cuda.is_available():
        y1 = [vectorized_func_gpu(batch_size) for batch_size in batch_sizes]
        y2 = [torch_func_gpu(batch_size) for batch_size in batch_sizes]

        plt.plot([batch_size * 1000 for batch_size in batch_sizes], y1, label="Torch Vectorized")
        plt.plot([batch_size * 1000 for batch_size in batch_sizes], y2, label="Torch")
        plt.xlabel('Batch size')
        plt.ylabel('Execution time (\u03BCs)')
        plt.title('SymEig exec. time (\u03BCs) vs batch size (K) (GPU)')
        plt.legend()
        plt.savefig("gpu_performance.png")
        plt.close()"""

    if torch.cuda.is_available():
        y1 = [vectorized_func(batch_size) for batch_size in batch_sizes]
        y2 = [vectorized_func_gpu(batch_size) for batch_size in batch_sizes]

        plt.plot([batch_size * 1000 for batch_size in batch_sizes], y1, label="Torch Vectorized (CPU)")
        plt.plot([batch_size * 1000 for batch_size in batch_sizes], y2, label="Torch Vectorized (GPU)")
        plt.xlabel('Batch size')
        plt.ylabel('Execution time (ms)')
        plt.title('Exec. time vs amount of eigendecomposition (GPU)')
        plt.legend()
        plt.savefig("performance_vectorized.png")
        plt.close()

