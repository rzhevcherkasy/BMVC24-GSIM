#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchvision import transforms
from PIL import Image
from utils.image_utils import erode
import numpy as np

def predicted_normal_loss(normal, normal_ref, alpha=None):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (3, H, W), normal_ref: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device) 
    else:
        weight = torch.ones_like(normal_ref)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    n = normal_ref.permute(1,2,0).reshape(-1,3).detach()
    n_pred = normal.permute(1,2,0).reshape(-1,3)
    loss = (w * (1.0 - torch.sum(n * n_pred, axis=-1))).mean()

    return loss

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    


def depth_smoothness_loss(inputs, weights, gamma):
    """
    Calculate depth smoothness loss based on the given criteria.

    Args:
    - gt_img (torch.Tensor): Ground truth RGB image with shape [H, W, 3].
    - pred_img (torch.Tensor): Predicted RGB image with shape [H, W, 3].
    - depth (torch.Tensor): Depth image with shape [H, W, 1].

    Returns:
    - torch.Tensor: Depth smoothness loss.
    """

    loss = lambda x: torch.mean(torch.abs(x))
    bilateral_filter = lambda x: torch.exp(-torch.abs(x).sum(-1) / gamma)
    w1 = bilateral_filter(weights[:,:-1] - weights[:,1:])
    w2 = bilateral_filter(weights[:-1,:] - weights[1:,:])
    w3 = bilateral_filter(weights[:-1,:-1] - weights[1:,1:])
    w4 = bilateral_filter(weights[1:,:-1] - weights[:-1,1:])

    w1 = w1.unsqueeze(-1)
    w2 = w2.unsqueeze(-1)
    w3 = w3.unsqueeze(-1)
    w4 = w4.unsqueeze(-1)

    L1 = loss(w1 * (inputs[:,:-1] - inputs[:,1:]))
    L2 = loss(w2 * (inputs[:-1,:] - inputs[1:,:]))
    L3 = loss(w3 * (inputs[:-1,:-1] - inputs[1:,1:]))
    L4 = loss(w4 * (inputs[1:,:-1] - inputs[:-1,1:]))

    return (L1 + L2 + L3 + L4) / 4 

def cosine_similarity_loss(normal_unmask, mask, num_samples=100):
    
    # Normalize the normals
    normals = normal_unmask[mask]
    normals = F.normalize(normals, p=2, dim=1)
    indices = torch.randperm(normals.size(0))[:num_samples]
    normals = normals[indices]

    # Calculate pairwise cosine similarity
    similarity_matrix = torch.matmul(normals, normals.t())
    
    # Diagonal elements represent the cosine similarity with itself, subtract 1
    similarity_loss = (similarity_matrix - torch.eye(similarity_matrix.size(0), device=normals.device)).pow(2).sum()

    # Normalize the loss by the number of pairs
    num_pairs = (similarity_matrix.size(0) * similarity_matrix.size(1) - similarity_matrix.size(0))
    similarity_loss /= num_pairs

    return 1-similarity_loss      

def mask_resize(img):
    resizer = transforms.Resize([60,80],interpolation=Image.NEAREST)
    mask_pil=transforms.functional.to_pil_image(img)
    img11 = resizer(mask_pil)
    #img11 = img11[None].cuda()
    img11 = transforms.functional.to_tensor(img11).cuda()
    img11 = torch.where(img11==0.,0,img11).detach()
    return img11


def loss_mse(reder_img,gt_img, masks):
    pred = (reder_img.permute(1,2,0)[masks]).view(-1,3)
    gt = (gt_img.permute(1,2,0)[masks]).view(-1,3).clone()
    loss_fn = torch.nn.MSELoss()
    loss2 = loss_fn(pred,gt)
    return loss2

def point_to_plane_distance(points, normal, depth):
    # Assuming points is a tensor of shape (M, 3)
    # normal and depth are tensors
    
    # Calculate the distance from each point to the plane
    distances = torch.abs(torch.matmul(points, normal)+ depth) 
    
    return distances

def plane_regularization_loss(points, normal, depth):
    # Calculate the point-to-plane distance
    distances = point_to_plane_distance(points, normal, depth)
    
    # Compute the mean squared distance
    loss = torch.mean(distances)
    
    return loss