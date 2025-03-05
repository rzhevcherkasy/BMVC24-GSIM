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
from icomma_diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image
import torch.nn.functional as F

def rendered_world2cam(viewpoint_cam, normal, alpha, bg_color):
    # normal: (3, H, W), alpha: (H, W), bg_color: (3)
    # normal_cam: (3, H, W)
    _, H, W = normal.shape
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()
    normal_world = normal.permute(1,2,0).reshape(-1, 3) # (HxW, 3)
    normal_cam = torch.cat([normal_world, torch.ones_like(normal_world[...,0:1])], axis=-1) @ torch.inverse(torch.inverse(extrinsic_matrix).transpose(0,1))[...,:3]
    normal_cam = normal_cam.reshape(H, W, 3).permute(2,0,1) # (H, W, 3)
    
    background = bg_color[...,None,None]
    normal_cam = normal_cam*alpha[None,...] + background*(1. - alpha[None,...])

    return normal_cam

def render_normal(viewpoint_cam, depth, bg_color, alpha):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()

    normal_ref = normal_from_depth_image(depth, intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device))
    background = bg_color[None,None,...]
    normal_ref = normal_ref*alpha[...,None] + background*(1. - alpha[...,None])
    normal_ref = normal_ref.permute(2,0,1)

    return normal_ref


def normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None,...]>0.).repeat(3, 1, 1)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)

def depth_alpha(depth, alpha):
    # depth: (3, H, W), alpha: (H, W)
    depth = depth.permute(1, 2, 0)
        # Expand alpha to match depth's shape
    alpha = alpha.unsqueeze(2).expand_as(depth)
    
    depth_im = torch.where(alpha > 0.01, depth/ alpha, depth.detach().max())
    return depth_im.permute(2, 0, 1)


def render(
        viewpoint_camera, 
        pc : GaussianModel,
        pipe, bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        override_color = None, 
        render_depth = True,
        render_normals = True,
        compute_grad_cov2d=True
    ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    This is a modified version where we can also get render depth_map and normal_map

    Partly follow 'Gaussian_Shader' paper
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        compute_grad_cov2d=compute_grad_cov2d,
        proj_k=viewpoint_camera.projection_matrix
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # (N, 3)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        camera_center = viewpoint_camera.camera_center,
        camera_pose = viewpoint_camera.world_view_transform)
    
    out_extras = None
    if render_depth:
        # Calculate Gaussians projected depth
        projvect1 = viewpoint_camera.world_view_transform[:,2][:3].detach()
        projvect2 = viewpoint_camera.world_view_transform[:,2][-1].detach()
        means3D_depth = (means3D * projvect1.unsqueeze(0)).sum(dim=-1,keepdim=True) + projvect2
        means3D_depth = means3D_depth.repeat(1,3)
        render_depth, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = means3D_depth,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            camera_center = viewpoint_camera.camera_center,
            camera_pose = viewpoint_camera.world_view_transform)
        render_depth = render_depth.mean(dim=0)

        render_extras = {}

        normal, delta_normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized, return_delta=True) # (N, 3)
        delta_normal_norm = delta_normal.norm(dim=1, keepdim=True)

        if render_normals:
            normal_unnormed = normal
            render_extras.update({"real_normal": normal_unnormed})
            normal_normed = 0.5*normal + 0.5  # range (-1, 1) -> (0, 1)
            render_extras.update({"normal": normal_normed})
            if delta_normal_norm is not None:
                render_extras.update({"delta_normal_norm": delta_normal_norm.repeat(1, 3)})
            

        out_extras = {}
        out_extras["depth"] = render_depth
        for k in render_extras.keys():
            if render_extras[k] is None: continue
            image = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = render_extras[k],
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp,
                camera_center = viewpoint_camera.camera_center.detach(),
                camera_pose = viewpoint_camera.world_view_transform.detach())[0]
            out_extras[k] = image        

        # Rasterize visible Gaussians to alpha mask image. 
        raster_settings_alpha = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            compute_grad_cov2d=compute_grad_cov2d,
            proj_k=viewpoint_camera.projection_matrix
        )
        rasterizer_alpha = GaussianRasterizer(raster_settings=raster_settings_alpha)
        alpha = torch.ones_like(means3D) 
        out_extras["alpha"] =  rasterizer_alpha(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = alpha,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            camera_center = viewpoint_camera.camera_center,
            camera_pose = viewpoint_camera.world_view_transform)[0]
        if render_normals:
            out_extras["normal_ref"] = render_normal(viewpoint_cam=viewpoint_camera, depth=out_extras['depth'], bg_color=bg_color, alpha=out_extras['alpha'][0])
            normalize_normal_inplace(out_extras["normal"], out_extras["alpha"][0])
        #out_extras["depth"] = depth_alpha(out_extras["depth"], out_extras["alpha"][0])
        


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii, 
    }
    if out_extras is not None:
        out.update(out_extras)
    return out


def render_virtual(
        viewpoint_camera_origin, 
        pc : GaussianModel, 
        pipe, bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        override_color = None, 
        compute_grad_cov2d=True,
        render_depth = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    viewpoint_camera = viewpoint_camera_origin
    c2w = torch.inverse(viewpoint_camera.world_view_transform.transpose(0, 1).cuda())
    R_ori = c2w[:3, :3]
    T_ori = c2w[:3, 3:4]
    I = torch.eye(3,device="cuda")
    M = I - 2 * torch.ger(pc.plane_normal,pc.plane_normal)
    R_ref = torch.matmul(M, R_ori)
    T_ref = torch.matmul(M, T_ori) - (2*pc.plane_depth* pc.plane_normal).unsqueeze(1)
    viewmat_ref = torch.eye(4, device=c2w.device, dtype= c2w.dtype)
    viewmat_ref[:3, :3] = R_ref
    viewmat_ref[:3, 3:4] = T_ref
    new_world_view_transform = torch.inverse(viewmat_ref).transpose(0, 1)
    new_camera_center = new_world_view_transform.inverse()[3, :3]
    new_projmatrix = (new_world_view_transform.unsqueeze(0).bmm(viewpoint_camera_origin.projection_matrix.detach().unsqueeze(0))).squeeze(0)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=new_world_view_transform,
        projmatrix=new_projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=new_camera_center,
        prefiltered=False,
        debug=pipe.debug,
        compute_grad_cov2d=compute_grad_cov2d,
        proj_k=viewpoint_camera.projection_matrix
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - new_camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # filter points
    normal_norm = pc.plane_normal
    
    filter_thres = 0.05
    distances = -torch.matmul(means3D, normal_norm)
    means3D =  means3D[distances <= pc.plane_depth-filter_thres].detach()
    means2D = means2D[distances <= pc.plane_depth-filter_thres].detach()
    shs = shs[distances <= pc.plane_depth-filter_thres].detach()
    if colors_precomp is not None:
        colors_precomp = colors_precomp[distances <= pc.plane_depth-filter_thres].detach()
    opacity = opacity[distances <= pc.plane_depth-filter_thres].detach()
    scales = scales[distances <= pc.plane_depth-filter_thres].detach()
    rotations = rotations[distances <= pc.plane_depth-filter_thres].detach()
    if cov3D_precomp is not None:
        cov3D_precomp = cov3D_precomp[distances <= pc.plane_depth-filter_thres].detach()

    if render_depth:
        # Calculate Gaussians projected depth
        projvect1 = new_world_view_transform[:,2][:3].detach()
        projvect2 = new_world_view_transform[:,2][-1].detach()
        means3D_depth = (means3D * projvect1.unsqueeze(0)).sum(dim=-1,keepdim=True) + projvect2
        means3D_depth = means3D_depth.repeat(1,3)
        render_depth, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = means3D_depth,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            camera_center = new_camera_center,
            camera_pose = new_world_view_transform)
        render_depth = render_depth.mean(dim=0)
        out_extras = {}
        out_extras["depth"] = render_depth

        intrinsic_matrix, _ = viewpoint_camera.get_calib_matrix_nerf()
        extrinsic_matrix = new_world_view_transform.transpose(0,1).contiguous()
        out_extras["intrinsic_matrix"] = intrinsic_matrix
        out_extras["extrinsic_matrix"] = extrinsic_matrix


    #Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        camera_center = new_camera_center,
        camera_pose = new_world_view_transform)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out =  {"render": rendered_image,
            "viewspace_points": screenspace_points.detach(),
            "visibility_filter" : radii > 0,
            "radii": radii}
    if out_extras is not None:
        out.update(out_extras)
    return out


def render_virtual_final(
        viewpoint_camera_origin, 
        pc : GaussianModel, 
        pipe, bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        override_color = None, 
        compute_grad_cov2d=True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    viewpoint_camera = viewpoint_camera_origin
    c2w = torch.inverse(viewpoint_camera.world_view_transform.transpose(0, 1).cuda())
    R_ori = c2w[:3, :3]
    T_ori = c2w[:3, 3:4]
    I = torch.eye(3,device="cuda")
    M = I - 2 * torch.ger(pc.plane_normal,pc.plane_normal)
    R_ref = torch.matmul(M, R_ori)
    T_ref = torch.matmul(M, T_ori) - (2*pc.plane_depth* pc.plane_normal).unsqueeze(1)
    viewmat_ref = torch.eye(4, device=c2w.device, dtype= c2w.dtype)
    viewmat_ref[:3, :3] = R_ref
    viewmat_ref[:3, 3:4] = T_ref
    new_world_view_transform = torch.inverse(viewmat_ref).transpose(0, 1)
    new_camera_center = new_world_view_transform.inverse()[3, :3]
    new_projmatrix = (new_world_view_transform.unsqueeze(0).bmm(viewpoint_camera_origin.projection_matrix.detach().unsqueeze(0))).squeeze(0)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=new_world_view_transform,
        projmatrix=new_projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=new_camera_center,
        prefiltered=False,
        debug=pipe.debug,
        compute_grad_cov2d=compute_grad_cov2d,
        proj_k=viewpoint_camera.projection_matrix
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - new_camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # filter points
    normal_norm = pc.plane_normal
    

    filter_thres = 0.01 #0.01
    distances = -torch.matmul(means3D, normal_norm)
    means3D =  means3D[distances <= pc.plane_depth-filter_thres]
    means2D = means2D[distances <= pc.plane_depth-filter_thres]
    shs = shs[distances <= pc.plane_depth-filter_thres]
    if colors_precomp is not None:
        colors_precomp = colors_precomp[distances <= pc.plane_depth-filter_thres]
    opacity = opacity[distances <= pc.plane_depth-filter_thres]
    scales = scales[distances <= pc.plane_depth-filter_thres]
    rotations = rotations[distances <= pc.plane_depth-filter_thres]
    if cov3D_precomp is not None:
        cov3D_precomp = cov3D_precomp[distances <= pc.plane_depth-filter_thres]

    #Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        camera_center = new_camera_center,
        camera_pose = new_world_view_transform)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points.detach(),
            "visibility_filter" : radii > 0,
            "radii": radii}
