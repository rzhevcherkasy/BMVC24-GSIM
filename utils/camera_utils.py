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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, CVtoTorchMask
from utils.graphics_utils import fov2focal
from typing import Dict, List, Optional, Tuple, Type, Union
import torch
from tqdm import tqdm
from splatfacto_config import SplatfactoModelConfig
import pyransac3d as pyrsc
WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 800
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    # if resized_image_rgb.shape[1] == 4:
    #     loaded_mask = resized_image_rgb[3:4, ...]
    loaded_mask = CVtoTorchMask(cam_info.masks, resolution)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, cx=int(cam_info.cx / scale), cy=int(cam_info.cy / scale), cx_ori=int(cam_info.cx_ori / scale), cy_ori=int(cam_info.cy_ori / scale),
                  image=gt_image, mask=loaded_mask,
                  image_name=cam_info.image_name, image_path = cam_info.image_path, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    # for id, c in enumerate(cam_infos):
    #     camera_list.append(loadCam(args, id, c, resolution_scale))
    with tqdm(enumerate(cam_infos), total=len(cam_infos)) as t:
        for id, c in t:
            t.set_description("{}".format(c.image_name))
            camera_list.append(loadCam(args, id, c, resolution_scale))
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {   
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def get_camera_coords(img_size: tuple, pixel_offset: float = 0.5) -> torch.Tensor:
    """Generates camera pixel coordinates [W,H]
    Returns:
        stacked coords [H*W,2] where [:,0] corresponds to W and [:,1] corresponds to H
    """
    # img size is (w,h)
    image_coords = torch.meshgrid(
        torch.arange(img_size[0]),
        torch.arange(img_size[1]),
        indexing="xy",  # W = u by H = v
    )
    image_coords = (
        torch.stack(image_coords, dim=-1) + pixel_offset
    )  # stored as (x, y) coordinates
    image_coords = image_coords.view(-1, 2)
    image_coords = image_coords.float()
    return image_coords

def get_means3d_backproj(
    depths: torch.Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    c2w: torch.Tensor,
    device: torch.device,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List]:
    """Backprojection using camera intrinsics and extrinsics
    image_coords -> (x,y,depth) -> (X, Y, depth)
    Returns:
        Tuple of (means: Tensor, image_coords: Tensor)
    """
    if depths.dim() == 3:
        #depths = depths[...,0].unsqueeze(-1)
        depths = depths.view(-1, 3)
    elif depths.shape[-1] != 1:
        depths = depths.unsqueeze(-1).contiguous()
        depths = depths.view(-1, 1)
    if depths.dtype != torch.float:
        depths = depths.float()
        c2w = c2w.float()
    if c2w.device != device:
        c2w = c2w.to(device)
    image_coords = get_camera_coords(img_size)
    image_coords = image_coords.to(device)  # note image_coords is (H,W)
    # TODO: account for skew / radial distortion
    means3d = torch.empty(
        size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
    ).view(-1, 3)
    means3d[:, 0] = (image_coords[:, 0] - cx) * depths[:, 0] / fx  # x
    means3d[:, 1] = (image_coords[:, 1] - cy) * depths[:, 0] / fy  # y
    means3d[:, 2] = depths[:, 0]  # z
    if mask is not None:
        mask = mask.view(-1,)
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        means3d = means3d[mask]
        image_coords = image_coords[mask]
    if c2w is None:
        c2w = torch.eye((means3d.shape[0], 4, 4), device=device)
    # to world coords
    means3d = means3d @ torch.linalg.inv(c2w[..., :3, :3]) + c2w[..., :3, 3]
    return means3d, image_coords

def fit_plane_pyransac(points, normals, max_iterations=80000, threshold_distance=0.05):
    points = points.detach().cpu().numpy()
    normals = normals.detach().cpu().numpy()
    plane = pyrsc.Plane()
    best_eq, best_inliers = plane.fit(points, threshold_distance,max_iterations)
    best_plane_normal = best_eq[:3]
    best_plane_depth = best_eq[3]
    avg_normal = np.mean(normals[best_inliers], axis=0)
    if np.dot(best_plane_normal, avg_normal) < 0:
        best_plane_normal = -np.array(best_plane_normal)
        best_plane_depth = -best_plane_depth
    
    return (torch.tensor(best_plane_normal).cuda(), torch.tensor(best_plane_depth).cuda())


def fit_plane_ransac(points, normals, max_iterations=1000, threshold_distance=0.1):

    """
    Fits a plane to the input points and normals using RANSAC algorithm.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) containing world coordinates of points.
        normals (torch.Tensor): Tensor of shape (N, 3) containing normals corresponding to the points.
        max_iterations (int): Maximum number of RANSAC iterations. Default is 1000.
        threshold_distance (float): Threshold distance for considering a point as an inlier. Default is 0.1.

    Returns:
        tuple: A tuple containing:
            - best_plane (tuple): A tuple containing the normal vector and depth of the best-fit plane.
            - best_consensus_set (torch.Tensor): Indices of points forming the best-fit plane consensus set.
    """

    device = normals.device
    best_plane = None
    best_consensus_set = torch.tensor([], dtype=torch.long, device=device)
    
    for _ in range(max_iterations):
        # Step 1: Random Sampling
        sample_indices = torch.randperm(points.size(0), device=device)[:3]
        sampled_points = points[sample_indices]
        sampled_normals = normals[sample_indices]
        
        # Step 2: Model Fitting
        plane_normal = torch.mean(sampled_normals, dim=0)
        plane_depth = -torch.dot(plane_normal, sampled_points[0])  # Assuming points are given in world coordinates
        
        # Step 3: Outlier Detection
        distances = torch.abs(torch.matmul(points, plane_normal) + plane_depth) / torch.norm(plane_normal)
        consensus_set = torch.nonzero(distances < threshold_distance).squeeze()
        
        # Step 4: Consensus Set Selection
        if consensus_set.size(0) > best_consensus_set.size(0):
            best_consensus_set = consensus_set
            best_plane = (plane_normal, plane_depth)
    
    # Optionally, refit the model using all points in the best consensus set
    if best_consensus_set.size(0) >= 3:
        best_plane_points = points[best_consensus_set]
        best_plane_normals = normals[best_consensus_set]
        best_plane_normal = torch.mean(best_plane_normals, dim=0)
        best_plane_depth = -torch.dot(best_plane_normal, best_plane_points[0])
        best_plane = (best_plane_normal, best_plane_depth)
    
    return best_plane, best_consensus_set