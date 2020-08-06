What is Torch Vectorized ?
==========================
Batched and vectorized operations on volume of 3x3 symmetric matrices with Pytorch. The current Pytorch's implementation of batch eigen-decomposition is very slow when dealing with huge number of small matrices (e.g. 500k x 3x3). **This library offers some basic functions like vSymEig, vExpm and vLogm for fast computation (>250x faster) of huge number of small matrices with Pytorch using an analytical solution.**

vSymEig
=======
A quick closed-form solution for volumetric 3x3 matrices Eigen-Decomposition with Pytorch. Solves Eigen-Decomposition of data with shape Bx9xDxHxW, where B is the batch size, 9 is the flattened 3x3 symmetric matrices, D is the depth, H is the Height, W is the width. The goal is to accelerate the Eigen-Decomposition of multiple (>500k) small matrices (3x3) on GPU with Pytorch using an analytical solution.

.. image:: _static/vsymeig.png
  :width: 100%
  :alt: vSymeig

vExpm
=====
Based on vSymEig, computes the matrix exponential for batch of volumetric 3x3 matrices.

.. image:: _static/vexpm.png
  :width: 100%
  :alt: vExpm

vLogm
=====
Based on vSymEig, computes the matrix logarithm for batch of volumetric 3x3 matrices.

.. image:: _static/vlogm.png
  :width: 100%
  :alt: vLogm

Install me
==========

pip install torch-vectorized

How to use
==========

.. code-block:: python

    from torchvectorized.vlinalg import vSymeig

    # Random batch of volumetric 3x3 symmetric matrices of size 16x9x32x32x32
    input = sym(torch.rand(16, 9, 32, 32, 32))

    # Output eig_vals with size: 16x3x32x32x32 and eig_vecs with size 16,3,3,32,32,32
    eig_vals, eig_vecs = vSymeig(input, eigen_vectors=True)

    def sym(self, inputs):
        # Ensure symmetry of randomly generated 3x3 matrix using (X + X.T) / 2.0
        return (inputs + inputs[:, [0, 3, 6, 1, 4, 7, 2, 5, 8], :, :, :]) / 2.0



.. toctree::
  :maxdepth: 1
  :caption: Packages and Modules

  vlinalg
  nn

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`