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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render,render_virtual_final
import torchvision
from utils.general_utils import safe_state, PILtoTorch
from utils.image_utils import image_resize
from argparse import ArgumentParser
from PIL import Image
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(dataset, model_path, name, iteration, views, gaussians, pipeline, background, width, height):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if os.path.exists(os.path.join(dataset.source_path + "/images/" + view.image_name + ".png")):
            view.original_image = PILtoTorch(Image.open(dataset.source_path + "/images/" + view.image_name + ".png"), (view.image_width, view.image_height))
        else:
            view.original_image = PILtoTorch(Image.open(dataset.source_path + "/images/" + view.image_name + ".jpg"), (view.image_width, view.image_height))
        rendering = render(view, gaussians, pipeline, background)["render"]
        render_ref = render_virtual_final(view, gaussians, pipeline, background)["render"].permute(1, 2, 0).detach()
        gt_masks = view.masks.cuda().clone() > 0.5
        masks = gt_masks.squeeze(0).detach()
        pred_image = rendering.permute(1, 2, 0)
        pred_image[masks] = render_ref[masks]
        pred = pred_image.permute(2, 0, 1)
        gt = view.original_image[0:3, :, :]
        pred = image_resize(pred,(height,width))
        gt = image_resize(gt,(height,width))
        torchvision.utils.save_image(pred, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, width : int, height : int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset,dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, width, height)

        if not skip_test:
             render_set(dataset,dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, width, height)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--width",type=int)
    parser.add_argument("--height",type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.width, args.height)