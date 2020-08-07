# Torch Vectorized
> Batched and vectorized operations on volume of 3x3 symmetric matrices with Pytorch. The current Pytorch's implementation of batch eigen-decomposition is very slow when dealing with huge number of small matrices (e.g. 500k x 3x3). **This library offers some basic functions like vSymEig, vExpm and vLogm for fast computation (>250x faster) of huge number of small matrices with Pytorch using an analytical solution.**

#### Read the documentaton [HERE](https://torch-vectorized.readthedocs.io/en/latest/)

## vSymEig
> A quick closed-form solution for volumetric 3x3 matrices Eigen-Decomposition with Pytorch. Solves Eigen-Decomposition of data with shape Bx9xDxHxW, where B is the batch size, 9 is the flattened 3x3 symmetric matrices, D is the depth, H is the Height, W is the width. The goal is to accelerate the Eigen-Decomposition of multiple (>500k) small matrices (3x3) on GPU with Pytorch using an analytical solution.   

<img src="https://raw.githubusercontent.com/banctilrobitaille/torch-vectorized/master/icons/vsymeig.png" width="100%" vertical-align="bottom">

## vExpm
> Based on vSymEig, computes the matrix exponential for batch of volumetric 3x3 matrices.

<img src="https://raw.githubusercontent.com/banctilrobitaille/torch-vectorized/master/icons/vexpm.png" width="100%" vertical-align="bottom">

## vLogm
> Based on vSymEig, computes the matrix logarithm for batch of volumetric 3x3 matrices.

<img src="https://raw.githubusercontent.com/banctilrobitaille/torch-vectorized/master/icons/vlogm.png" width="100%" vertical-align="bottom">

## Install me

> pip install torch-vectorized

## How to use

```python
import torch
from torchvectorized.utils import sym
from torchvectorized.vlinalg import vSymEig

# Random batch of volumetric 3x3 symmetric matrices of size 16x9x32x32x32
input = sym(torch.rand(16, 9, 32, 32, 32))

# Output eig_vals with size: 16x3x32x32x32 and eig_vecs with size 16,3,3,32,32,32
eig_vals, eig_vecs = vSymEig(input, eigen_vectors=True)
```

## Contributing

#### How to contribute ?
- [X] Create a branch by feature and/or bug fix
- [X] Get the code
- [X] Commit and push
- [X] Create a pull request

#### Branch naming

##### Feature branch
> feature/ [Short feature description] [Issue number]

##### Bug branch
> fix/ [Short fix description] [Issue number]

#### Commits syntax:

##### Adding code:
> \+ Added [Short Description] [Issue Number]

##### Deleting code:
> \- Deleted [Short Description] [Issue Number]

##### Modifying code:
> \* Changed [Short Description] [Issue Number]

##### Merging code:
> Y Merged [Short Description] [Issue Number]


Icons made by <a href="http://www.flaticon.com/authors/freepik" title="Freepik">Freepik</a> from <a href="http://www.flaticon.com" title="Flaticon">www.flaticon.com</a> is licensed by <a href="http://creativecommons.org/licenses/by/3.0/" title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a>
