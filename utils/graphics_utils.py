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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt), np.float32(C2W)

def getProjectionMatrix(znear, zfar, fovX, fovY, cx, cy):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    # P[0, 2] = (right + left) / (right - left)
    # P[1, 2] = (top + bottom) / (top - bottom)
    P[0, 2] = cx
    P[1, 2] = cy
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz

def depth2point_cam(sampled_depth, ref_intrinsic):
    ref_intrinsic = ref_intrinsic.cuda()
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz

def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
    #xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[..., 0:1])], axis=-1) @ extrinsic_matrix.transpose(0, 1)
    #import pdb; pdb.set_trace()
    xyz_world = xyz_world[...,:3]

    return xyz_world

# def depth2point_world(depth: torch.Tensor, intrinsic: torch.Tensor, extrinsic: torch.Tensor):
#     """
#     Convert depth map to world coordinates.
    
#     Args:
#         depth (torch.Tensor): Depth map of shape (H, W)
#         intrinsic (torch.Tensor): Camera intrinsic matrix of shape (3,3)
#         extrinsic (torch.Tensor): Camera extrinsic matrix of shape (4,4)
    
#     Returns:
#         torch.Tensor: 3D points in world coordinates of shape (N, 3)
#     """
#     H, W = depth.shape
    
#     # Create a mesh grid of pixel coordinates
#     y, x = torch.meshgrid(torch.arange(H, device=depth.device), 
#                           torch.arange(W, device=depth.device), 
#                           indexing='ij')
#     xy_h = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1).cuda()  # (3, H*W)
    
#     # Flatten depth and filter out zero depth values
#     depth_flat = depth.reshape(-1)
#     valid_mask = depth_flat > 0
#     xy_h = xy_h[:, valid_mask]
#     depth_valid = depth_flat[valid_mask]
    
#     intrinsic = intrinsic.to(torch.float32)
#     xy_h = xy_h.to(torch.float32)

#     # Compute camera coordinates
#     cam_points = torch.linalg.solve(intrinsic.cuda(), xy_h.cuda()) * depth_valid.cuda().reshape(1, -1) 
    
#     # Convert to homogeneous coordinates (4, N)
#     cam_points_h = torch.cat([cam_points, torch.ones((1, cam_points.shape[1]), device=depth.device)], dim=0)
    
#     # Transform to world coordinates
#     world_points_h = extrinsic @ cam_points_h  # (4, N)
#     world_points = world_points_h[:3].T  # (N, 3)
    
#     return world_points


def depth_pcd2normal(xyz):
    hd, wd, _ = xyz.shape 
    bottom_point = xyz[..., 2:hd,   1:wd-1, :]
    top_point    = xyz[..., 0:hd-2, 1:wd-1, :]
    right_point  = xyz[..., 1:hd-1, 2:wd,   :]
    left_point   = xyz[..., 1:hd-1, 0:wd-2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point 
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2,0,1), (1,1,1,1), mode='constant').permute(1,2,0)
    return xyz_normal

def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    xyz_world = depth2point_world(depth, intrinsic_matrix, extrinsic_matrix) # (HxW, 3)
    xyz_world = xyz_world.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_world)

    return xyz_normal

def get_dtu_raydir(pixelcoords, intrinsic, rot, dir_norm):
    # rot is c2w
    ## pixelcoords: H x W x 2
    x = (pixelcoords[..., 0] + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (pixelcoords[..., 1] + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
    z = torch.ones_like(x)
    dirs = torch.stack([x, y, z], axis=-1)
    dirs = dirs @ rot[:,:].T #\
    if dir_norm:
        dirs = torch.nn.functional.normalize(dirs, dim=-1)
    return dirs

def get_rays(width, height, intrinsic, camrot):
    px, py = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32))

    pixelcoords = torch.stack((px, py), dim=-1).cuda()  # H x W x 2
    raydir = get_dtu_raydir(pixelcoords, intrinsic, camrot, dir_norm=True)
    return raydir