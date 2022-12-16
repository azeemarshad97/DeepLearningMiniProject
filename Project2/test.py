import torch
import math
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Generate train and test sets of 1000 points sampled uniformly in [0,1]^2, each with a label 0 if outside the disk centered at (0.5, 0.5) of radius 1/sqrt(2*pi), and 1 inside.
    train_set = torch.utils.data.TensorDataset(
        torch.rand(1000, 2), 
        torch.ones(1000)
        )

    print(train_set.shape)
    print(train_set)
    # plt.scatter(train_set[0][:,0], train_set[0][:,1])
    # plt.show()