from scipy.linalg import sqrtm
import numpy as np

import torch
import scipy


class MatrixSquareRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Ensure the input is a square matrix
        assert input.shape[0] == input.shape[1], "Input must be a square matrix"
        
        # Use the Newton-Schulz iteration to approximate the square root
        max_iter = 5
        eye = torch.eye(input.shape[0], device=input.device, dtype=input.dtype)
        Y = input
        Z = eye
        for i in range(max_iter):
            Y = 0.5 * (Y + torch.inverse(Z) @ input)
            Z = 0.5 * (Z + torch.inverse(Y) @ input)
        
        ctx.save_for_backward(Y, Z)
        return Y
    
    @staticmethod
    def backward(ctx, grad_output):
        Y, Z = ctx.saved_tensors
        grad_input = grad_output @ torch.inverse(Y).t() @ torch.inverse(Y).t()
        return grad_input


def matrix_sqrt(input):
    return MatrixSquareRoot.apply(input)


def calculate_fid(tensor_A, tensor_B):
    tensor_A = tensor_A.type(torch.float32)
    tensor_B = tensor_B.type(torch.float32)

    mu_A = tensor_A.mean(dim=0)
    mu_B = tensor_B.mean(dim=0)

    tensor_A_centered = tensor_A - mu_A
    tensor_B_centered = tensor_B - mu_B
    cov_A = tensor_A_centered.T @ tensor_A_centered / (tensor_A.size(0) - 1)
    cov_B = tensor_B_centered.T @ tensor_B_centered / (tensor_B.size(0) - 1)

    cov_A_np = cov_A.cpu().numpy()
    cov_B_np = cov_B.cpu().numpy()
    covmean = sqrtm(cov_A_np.dot(cov_B_np))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    mean_diff = mu_A - mu_B
    mean_diff_np = mean_diff.cpu().numpy()
    fid = np.sum(mean_diff_np**2) + np.trace(cov_A_np) + np.trace(cov_B_np) - 2 * np.trace(covmean)
    return fid


@torch.no_grad()
def calculate_fid_pytorch(tensor_A, tensor_B):
    mu_A = tensor_A.mean(dim=0)
    mu_B = tensor_B.mean(dim=0)

    tensor_A_centered = tensor_A - mu_A
    tensor_B_centered = tensor_B - mu_B
    cov_A = tensor_A_centered.T @ tensor_A_centered / (tensor_A.size(0) - 1)
    cov_B = tensor_B_centered.T @ tensor_B_centered / (tensor_B.size(0) - 1)

    covmean = matrix_sqrt(cov_A @ cov_B)

    if torch.is_complex(covmean):
        covmean = covmean.real

    mean_diff = mu_A - mu_B
    fid = torch.sum(mean_diff**2) + torch.trace(cov_A) + torch.trace(cov_B) - 2 * torch.trace(covmean)
    return fid.item()


def replace_outliers_with_median_l2(data):
    norms = torch.norm(data, p=2, dim=-1)  # Compute L2 norms along the last dimension
    median_norm = torch.median(norms)
    std_dev = torch.std(norms)
    outliers = torch.abs(norms - median_norm) > 3 * std_dev  # Outliers based on norms

    median_values = torch.median(data, dim=0).values
    data[outliers, :] = median_values
    return data


def replace_outliers_with_iqr_l2(data):
    norms = torch.norm(data, p=2, dim=1)
    Q1 = torch.quantile(norms, 0.25)
    Q3 = torch.quantile(norms, 0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outliers
    outliers = (norms < lower_bound) | (norms > upper_bound)
    
    # Replace whole rows with the median of the dataset
    median_values = torch.median(data, dim=0).values
    data[outliers, :] = median_values
    return data