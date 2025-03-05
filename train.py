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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, depth_smoothness_loss, cosine_similarity_loss, loss_mse,predicted_normal_loss
from gaussian_renderer import render, network_gui,render_virtual, render_virtual_final
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, PILtoTorch
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, iComMaParams
from utils.camera_utils import get_means3d_backproj, fit_plane_ransac, fit_plane_pyransac
from utils.graphics_utils import fov2focal,depth2point_world
import torch.optim as optim
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
import trimesh
import math
import PIL.Image as pil
from utils.vis_utils import convert_array_to_pil
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from splatfacto_config import SplatfactoModelConfig
import torchvision
from PIL import Image


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(6666)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    icommaparams = iComMaParams(parser)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 0]
    # bg_color = [1, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    #background = torch.rand((3), device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    optimizer_mirror = optim.Adam([gaussians.plane_normal,gaussians.plane_depth],lr = icommaparams.camera_pose_lr)
    blue_fall_cnt = 0
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        torch.cuda.synchronize()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if SplatfactoModelConfig.predict_normals:
            gaussians.set_requires_grad("normal", state=iteration >= SplatfactoModelConfig.stage_two_step)
            gaussians.set_requires_grad("normal2", state=iteration >= SplatfactoModelConfig.stage_two_step)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Save GPU memory
        viewpoint_cam.original_image = PILtoTorch(Image.open(viewpoint_cam.image_path), (viewpoint_cam.image_width, viewpoint_cam.image_height))       
        gt_image = viewpoint_cam.original_image.cuda().clone()
        gt_mask = viewpoint_cam.masks.cuda().clone()
        
        # Stage 1 :
        # In stage 1 we try to inirialize the whole scene
        if iteration <= SplatfactoModelConfig.stage_two_step:
            # render image
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_normals = False)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            loss.backward()

        # stage 2 :
        # In stage 2 we start to render depth and normal, and add regularizations
        elif iteration <= SplatfactoModelConfig.stage_three_step:

            masks = gt_mask
            # render image 
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # mask region to blue
            gt_image = gt_image.permute(1, 2, 0)
            gt_image[masks] = torch.tensor([0, 0, 255], dtype=gt_image.dtype, device="cuda")
            gt_image = gt_image.permute(2, 0, 1)
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            # add regularizations
            loss+= depth_smoothness_loss(render_pkg["depth"].unsqueeze(2), gt_image.permute(1, 2, 0),0.1)* SplatfactoModelConfig.depth_smooth_weight
            loss+= cosine_similarity_loss(render_pkg["normal"].permute(1, 2, 0), masks)* SplatfactoModelConfig.normal_smooth_weight
            loss+= predicted_normal_loss(render_pkg["normal"], render_pkg["normal_ref"].detach(), render_pkg["alpha"].detach())* SplatfactoModelConfig.predicted_normal_weight

            loss.backward()

        # stage 3 :
        # In stage 3 we select a random view that contains large mirror region, and used the render depth&normal map to initilize the mirror plane
        # Then we start to optimize the mirror plane
        elif iteration <= SplatfactoModelConfig.stage_four_step:
            masks = gt_mask
            blue_cnt_thres = 300
            if gaussians.construct:
                if blue_fall_cnt<= blue_cnt_thres:
                    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_normals = False)
                else:
                    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_depth= False, render_normals= False)
            else:
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            loss = None

            # Plane already constructed 
            if gaussians.construct: 
                # before mirror plane optimization, we firstly need to make sure the blue 3D-GS in fornt of or on the mirror plane shall not affect virtual rendering
                # This is a naive but bad solution to just push the blue gs behind the mirror 
                # This can be replaced by some better solutions: depth supversion or let them fall on the estimated plane
                if blue_fall_cnt<= blue_cnt_thres:
                    depth = render_pkg["depth"].unsqueeze(2)
                    depth_gt = depth.clone()
                    depth_gt[masks] = torch.tensor([24], dtype=gt_image.dtype, device="cuda")
                    L_pseudo_depth = torch.abs(depth[masks]-depth_gt[masks]).mean()
                    loss = L_pseudo_depth
                    blue_fall_cnt +=1

                else:
                    image = render_pkg["render"]
                    gt_image = gt_image.permute(1, 2, 0)
                    pred_image = image.permute(1, 2, 0)
                    Ll1 = torch.abs(gt_image[~masks] - pred_image[~masks]).mean()

                    gt_image_sim = gt_image.clone()
                    pred_image_sim = pred_image.clone()
                    gt_image_sim[masks] = torch.tensor([0, 0, 255], dtype=gt_image.dtype, device="cuda")
                    pred_image_sim[masks] = torch.tensor([0, 0, 255], dtype=gt_image.dtype, device="cuda")
                    L_ssim = ssim(pred_image_sim.permute(2, 0, 1), gt_image_sim.permute(2, 0, 1))
                    gt_rgb_ref = viewpoint_cam.original_image.cuda().clone()


                    # Here when doing virtual rendering, we freeze the 3D-GS properties and only optimize the mirror plane
                    if masks.sum()>1000:
                        render_pkg_ref = render_virtual(viewpoint_cam, gaussians, pipe, bg, render_depth = True)
                        ref_rgb, ref_dpeth, intrinsic, extrinsic = render_pkg_ref["render"], render_pkg_ref["depth"], render_pkg_ref["intrinsic_matrix"], render_pkg_ref["extrinsic_matrix"]

                        gt_rgb_ref = gt_rgb_ref.permute(1, 2, 0)
                        gt_rgb_ref[~masks] = torch.tensor([0, 0, 255], dtype=gt_rgb_ref.dtype, device="cuda")
                        gt_rgb_ref = gt_rgb_ref.permute(2, 0, 1)

                        ref_rgb = ref_rgb.permute(1, 2, 0)
                        ref_rgb[~masks] = torch.tensor([0, 0, 255], dtype=gt_rgb_ref.dtype, device="cuda")
                        ref_rgb = ref_rgb.permute(2, 0, 1)

                        loss_color = loss_mse(ref_rgb , gt_rgb_ref, masks)
                        loss_mirror =  loss_color
                        loss_mirror +=  (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - L_ssim)
                        new_lrate = icommaparams.camera_pose_lr 
                        for param_group in optimizer_mirror.param_groups:
                            param_group['lr'] = new_lrate
                    
                        optimizer_mirror.zero_grad()
                        loss_mirror.backward()
                        print(gaussians.plane_normal)
                        print(gaussians.plane_depth)
                        optimizer_mirror.step()
                        new_plane_normal = F.normalize(gaussians.plane_normal.clone().detach(),dim=-1)
                        gaussians.plane_normal.data.copy_(new_plane_normal)

            # plane unconstruct, so we need to estimate the mirror plane
            else:
                mirror_normal = render_pkg["real_normal"].permute(1, 2, 0)
                filtered_normals = mirror_normal[masks].view(-1,3)
                count = masks.sum()  
                magnitudes = torch.norm(filtered_normals, dim=1, keepdim=True)             
                filtered_normals = filtered_normals/ magnitudes
                if count > 30000:
                    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()
                    xyz_world = depth2point_world(render_pkg["depth"].clone().detach(), intrinsic_matrix, extrinsic_matrix) # (HxW, 3)
                    xyz_world = xyz_world.reshape(*render_pkg["depth"].clone().detach().shape, 3)
                    xyz_world = xyz_world.view(-1,3)
                    mask = masks.view(-1,)
                    means3d = xyz_world[mask]
                    normal_mirror = filtered_normals
                    # Pyransac is much more robust
                    if SplatfactoModelConfig.py_ransac:
                        best_plane = fit_plane_pyransac(
                            points = means3d, 
                            normals = normal_mirror ,
                            threshold_distance= SplatfactoModelConfig.ransac_threshold
                        )
                    else:
                        best_plane, best_consensus_set = fit_plane_ransac(
                            points = means3d, 
                            normals = normal_mirror 
                        )
                    gaussians.plane_normal.data.copy_(best_plane[0])
                    gaussians.plane_depth.data.copy_(best_plane[1])
                    gaussians.construct = True

            if loss is not None:
                loss.backward()

        # stage 4 :
        else :
            # In this stage we used the fused rendering, and again start to optimize the 3D-GS properties
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_depth= False, render_normals= False)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            masks = gt_mask
            pred_image = image.permute(1, 2, 0)
            render_pkg_ref = render_virtual_final(viewpoint_cam, gaussians, pipe, bg)
            rendering = render_pkg_ref["render"].permute(1, 2, 0)
            pred_image[masks] = rendering[masks]
            pred = pred_image.permute(2, 0, 1)
            Ll1 = l1_loss(pred, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(pred, gt_image))
            new_lrate = icommaparams.camera_pose_lr_final
            for param_group in optimizer_mirror.param_groups:
                param_group['lr'] = new_lrate

            optimizer_mirror.zero_grad()
            loss.backward()
            optimizer_mirror.step()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Progress bar
            if loss is not None:
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            else:
                ema_loss_for_log = ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(dataset,tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and loss is not None:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 10 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            viewpoint_cam.original_image = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(dataset,tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    viewpoint.original_image = PILtoTorch(Image.open(viewpoint.image_path), (viewpoint.image_width, viewpoint.image_height))
                    # if SplatfactoModelConfig.syn_data:
                    #     viewpoint.original_image = PILtoTorch(Image.open(dataset.source_path + "/images/" + viewpoint.image_name+'.png'), (viewpoint.image_width, viewpoint.image_height))
                    # else:
                    #     viewpoint.original_image = PILtoTorch(Image.open(dataset.source_path + "/images/" + viewpoint.image_name+".png"), (viewpoint.image_width, viewpoint.image_height))
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    viewpoint.original_image = None
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[ 1000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 20_000, 30_000,40_000,50_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
