import torch

def tv_l1_regularization(x):
    """
    Total variation regularization for 2D image
    :param x: Input image
    :return: Total variation regularization
    """
    
    x_diff = x[:, 1:] - x[:, :-1]
    y_diff = x[1:, :] - x[:-1, :]
    return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))

def tv_l2_regularization(x):
    """
    Total variation regularization for 2D image
    :param x: Input image
    :return: Total variation regularization
    """
    
    x_diff = x[:, 1:] - x[:, :-1]
    y_diff = x[1:, :] - x[:-1, :]
    return torch.sum(torch.square(x_diff)) + torch.sum(torch.square(y_diff))

def laplacian_regularization(x):
    """
    Laplacian regularization for 2D image
    :param x: Input image
    :return: Laplacian regularization
    """
    
    x_diff = x[:, 1:] - x[:, :-1]
    y_diff = x[1:, :] - x[:-1, :]
    return torch.sum(torch.square(x_diff[:, 1:] - x_diff[:, :-1])) + torch.sum(
        torch.square(y_diff[1:, :] - y_diff[:-1, :])
    )

def tikhonov_regularization(x):
    """
    Tikhonov regularization for 2D image
    :param x: Input image
    :return: Total variation regularization
    """
    
    x_diff = x[:, 1:] - x[:, :-1]
    y_diff = x[1:, :] - x[:-1, :]
    return torch.sum(torch.square(x_diff)) + torch.sum(torch.square(y_diff))


def second_order_tikhonov_regularization(x):
    """
    Second order Tikhonov regularization for 2D image
    :param x: Input image
    :return: Tikhonov 2nd-order regularization
    """
    
    x_diff = x[:, 1:] - x[:, :-1]
    y_diff = x[1:, :] - x[:-1, :]
    return torch.sum(torch.square(x_diff[:, 1:] - x_diff[:, :-1])) + torch.sum(
        torch.square(y_diff[1:, :] - y_diff[:-1, :])
    )

def l1_regularization(x):
    """
    L1 regularization for 2D image
    :param x: Input image
    :return: L1 regularization
    """
    
    return torch.sum(torch.abs(x))

def l2_regularization(x):
    """
    L2 regularization for 2D image
    :param x: Input image
    :return: L2 regularization
    """
    
    return torch.sum(torch.square(x))