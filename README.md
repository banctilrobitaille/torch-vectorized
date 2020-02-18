# Volumetric Eigen-Decomposition of 3x3 matrices
> A quick closed-form solution for volumetric 3x3 matrices Eigen-Decomposition with Pytorch. Solves Eigen-Decomposition of data with shape Bx9xDxHxW, where B is the batch size, 9 is the flattened 3x3 symmetric matrices, D is the depth, H is the Height, W is the width. The goal is to accelerate the Eigen-Decomposition of multiple (>500k) small matrices (3x3) on GPU with Pytorch using an analytical solution.   

## Install me

> pip install torch-vectorized

## How to use

```python
from torchvectorized.vlinalg import vSymeig

# Random batch of volumetric 3x3 symmetric matrices of size 16x9x32x32x32
input = sym(torch.rand(16, 9, 32, 32, 32))

# Output eig_vals with size: 16x3x32x32x32 and eig_vecs with size 16,3,3,32,32,32
eig_vals, eig_vecs = vSymeig(input, eigen_vectors=True)

def sym(self, inputs):
    # Ensure symmetry of randomly generated 3x3 matrix using (X + X.T) / 2.0
    return (inputs + inputs[:, [0, 3, 6, 1, 4, 7, 2, 5, 8], :, :, :]) / 2.0
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
