import math
import numpy as np
import ot as pot
import torch

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def wasserstein(source_data, target_data, p=1):
    '''
        Compute the 1 and 2 Wasserstein distances
        See https://pythonot.github.io/quickstart.html#optimal-transport-and-wasserstein-distance
        
        Parameter
        ---------
        source_data:    Tensor, shape [batch_size, dim]
        target_data:    Tensor, shape [batch_size, dim]
        p:              int, 
                            p-Wasserstein distance, only supports 1 and 2

        Returns
        -------
        float
            the p-Wasserstein distance
    '''


    assert p in [1, 2], "only support 1 and 2 Wasserstein distances!"
    source_data = source_data.to(device)
    target_data = target_data.to(device)
    source_sample_weights =  torch.ones(source_data.shape[0]) / source_data.shape[0]
    target_sample_weights =  torch.ones(target_data.shape[0]) / target_data.shape[0]
    if p == 1:
        M = pot.dist(source_data, target_data, metric="euclidean")
        W = pot.emd2(source_sample_weights, target_sample_weights, M).item()
    else:
        M = pot.dist(source_data, target_data) # metric is squared euclidean distance by default
        W = math.sqrt(pot.emd2(source_sample_weights, target_sample_weights, M))
    return W


def gaussian_kernel_matrix(X, Y, sigma=1.0, zero_diagonal=False):
    """
        Compute the Gaussian kernel matrix given data X and Y and sigma
        The Gaussian kernel matrix is given by
            K_ij = exp(-||x_i-y_j||^2/(2*sigma^2))
        
        Parameters
        ----------
        X:      Tensor, shape [n_X, dim]
        Y:      Tensor, shape [n_Y, dim]
        sigma:  Float
        zero_diagonal: Boolean, whether we want to zero out the diagonal
        
        Returns
        -------
        Tensor,
            shape [n_X, n_Y], where the (i,j)th entry is given as above
    """
    square_eucl_dist = torch.cdist(X, Y, p=2) ** 2
    # Compute the Gaussian kernel matrix
    gamma = 1.0 / (2 * sigma ** 2)
    K = torch.exp(-gamma * square_eucl_dist)
    if zero_diagonal:
        K.fill_diagonal_(0)
    return K

def rbf_mmd(X, Y, sigma_list=[0.01, 0.1, 1, 10, 100]):
    """
        Compute the mean Maximum Mean Discrepancy (MMD) with Gaussian kernel, with different sigmas,
        between two data set.
        
        Parameters
        ----------
        X:      Tensor, shape [batch_size, dim]
        Y:      Tensor, shape [batch_size, dim]
        sigma:  List of floats
        
        Returns
        -------
        float,
            the rbf-MMD value averaged over sigma_list
    """
    X = X.to(device)
    Y = Y.to(device)
    n_X = X.shape[0]
    n_Y = Y.shape[0]
    MMD_list = []
    
    for sigma in sigma_list:
        K_XX = gaussian_kernel_matrix(X, X, sigma, zero_diagonal = True)
        K_YY = gaussian_kernel_matrix(Y, Y, sigma, zero_diagonal = True)
        K_XY = gaussian_kernel_matrix(X, Y, sigma, zero_diagonal = False)

        XX_component = torch.sum(K_XX) / (n_X * (n_X - 1))
        YY_component = torch.sum(K_YY) / (n_Y * (n_Y - 1))
        XY_component = torch.sum(K_XY) / (n_X * n_Y)
        MMD = XX_component + YY_component - 2 * XY_component
        MMD_list.append(MMD.item())
    
    return sum(MMD_list)/len(MMD_list)

