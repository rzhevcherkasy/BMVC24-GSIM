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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, get_rays

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, cx, cy, cx_ori, cy_ori, image, mask,
                 image_name,image_path, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 normal_image=None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.cx = cx
        self.cy = cy
        self.cx_ori = cx_ori
        self.cy_ori = cy_ori
        self.image_name = image_name
        self.image_path = image_path

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        #self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.original_image = image.clamp(0.0, 1.0).to("cpu")
        self.masks = mask
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]


        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        w2c, c2w = getWorld2View2(R, T, trans, scale)
        self.w2c = torch.tensor(w2c).cuda()
        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=self.cx, cy=self.cy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.original_image = None
        self.c2w = torch.tensor(c2w).cuda()

    def get_calib_matrix_nerf(self):
        focalx = fov2focal(self.FoVx, self.image_width)  # original focal length
        focaly = fov2focal(self.FoVy, self.image_height)  # original focal length
        intrinsic_matrix = torch.tensor([[focalx, 0, self.cx_ori], [0, focaly, self.cy_ori], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix


    def get_rays(self):
        intrinsic_matrix, extrinsic_matrix = self.get_calib_matrix_nerf()

        viewdirs = get_rays(self.image_width, self.image_height, intrinsic_matrix, extrinsic_matrix[:3,:3])
        return viewdirs

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

