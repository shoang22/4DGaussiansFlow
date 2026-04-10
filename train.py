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
import numpy as np
import random
import os, sys
import torch
from random import gauss, randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui, render_flow, deform, final_from_deformation_delta, render_helper
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from utils.flow_utils import calculate_gs_flow, flow_loss, warping_gs_flow, calculate_camera_flow
from time import time
import copy
from gmflow.gmflow import build_gmflow
from gmflow.config import get_cfg as get_gmflow_cfg 

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians: GaussianModel, scene, stage, tb_writer, train_iter,timer):
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    ##### GMFlow
    cfg = get_gmflow_cfg()
    flownet = torch.nn.DataParallel(build_gmflow(cfg)) 
    flownet = flownet.module
    checkpoint = torch.load(cfg.model, map_location = 'cpu')
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    flownet.load_state_dict(weights)
    flownet = flownet.cuda()
    flownet.eval()

    flow_cache = {}  # cache GMFlow results keyed by (cam1.image_name, cam2.image_name)

    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        # TODO: viewpoint 1 and 2 is not implemented for not opt.dataloader
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    # 
    batch_size = opt.batch_size * 2
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 
                            # 
    count = 0
    for iteration in range(first_iter, final_iter+1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             count +=1
        #             viewpoint_index = (count ) % len(video_cams)
        #             if (count //(len(video_cams))) % 2 == 0:
        #                 viewpoint_index = viewpoint_index
        #             else:
        #                 viewpoint_index = len(video_cams) - viewpoint_index - 1
        #             # print(viewpoint_index)
        #             viewpoint = video_cams[viewpoint_index]
        #             custom_cam.time = viewpoint.time
        #             # print(custom_cam.time, viewpoint_index, count)
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]

        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive) :
        #             break
        #     except Exception as e:
        #         print(e)
        #         network_gui.conn = None

        iter_start.record()
        t_iter_start = time()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # dynerf's branch
        t_data_start = time()
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :

                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
        t_data_end = time()
        # print(len(viewpoint_cams))
        # breakpoint()
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        viewpoint_cams = list(viewpoint_cams)
        viewpoint_cams1 = viewpoint_cams[::2]
        viewpoint_cams2 = viewpoint_cams[1::2]
        t_forward_start = time()
        _fwd_timings = {"deform1": 0.0, "deform2": 0.0, "deform_delta": 0.0,
                        "render1": 0.0, "render2_1": 0.0, "render2": 0.0,
                        "gs_flow": 0.0, "render_coarse": 0.0}
        stage = "fine"
        for i in range(len(viewpoint_cams) // 2):
            viewpoint_cam1 = viewpoint_cams1[i]
            viewpoint_cam2 = viewpoint_cams2[i]

            means3D = gaussians.get_xyz
            opacity = gaussians._opacity
            shs = gaussians.get_features
            scales = gaussians._scaling
            rotations = gaussians._rotation
            loss = 0
            if "coarse" in stage:
                means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
                _t = time()
                render_pkg = render_flow(
                    viewpoint_camera=viewpoint_cam2,
                    pc=gaussians,
                    xyz=means3D_final,
                    scales=scales_final,
                    rotations=rotations_final,
                    opacity=opacity_final,
                    shs=shs_final,
                    pipe=pipe,
                    bg_color=background,
                    cam_type=scene.dataset_type
                )
                _fwd_timings["render_coarse"] += time() - _t
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            elif "fine" in stage:
                _t = time()
                dx1, ds1, dr1, do1, dshs1, mask1, scales_emb1, rotations_emb1 = deform(viewpoint_camera=viewpoint_cam1, pc=gaussians, means3D=means3D, scales=scales, rotations=rotations, opacity=opacity, shs=shs, cam_type=scene.dataset_type)
                _fwd_timings["deform1"] += time() - _t

                _t = time()
                dx2, ds2, dr2, do2, dshs2, mask2, scales_emb2, rotations_emb2 = deform(viewpoint_camera=viewpoint_cam2, pc=gaussians, means3D=means3D, scales=scales, rotations=rotations, opacity=opacity, shs=shs, cam_type=scene.dataset_type)
                _fwd_timings["deform2"] += time() - _t

                _t = time()
                means3D_final1, scales_final1, rotations_final1, opacity_final1, shs_final1 = final_from_deformation_delta(
                    viewpoint_camera=viewpoint_cam1,
                    pc=gaussians,
                    dx=dx1,
                    ds=ds1,
                    dr=dr1,
                    do=do1,
                    dshs=dshs1,
                    stage=stage,
                    precomputed=(mask1, scales_emb1, rotations_emb1),
                )
                means3D_final2_1, scales_final2_1, rotations_final2_1, opacity_final2_1, shs_final2_1 = final_from_deformation_delta(
                    viewpoint_camera=viewpoint_cam2,
                    pc=gaussians,
                    dx=dx1,
                    ds=ds1,
                    dr=dr1,
                    do=do1,
                    dshs=dshs1,
                    stage=stage,
                    precomputed=(mask2, scales_emb2, rotations_emb2),
                )
                means3D_final2, scales_final2, rotations_final2, opacity_final2, shs_final2 = final_from_deformation_delta(
                    viewpoint_camera=viewpoint_cam2,
                    pc=gaussians,
                    dx=dx2,
                    ds=ds2,
                    dr=dr2,
                    do=do2,
                    dshs=dshs2,
                    stage=stage,
                    precomputed=(mask2, scales_emb2, rotations_emb2),
                )
                _fwd_timings["deform_delta"] += time() - _t

                _t = time()
                render_pkg1 = render_flow(
                    viewpoint_camera=viewpoint_cam1,
                    pc=gaussians,
                    xyz=means3D_final1,
                    scales=scales_final1,
                    rotations=rotations_final1,
                    opacity=opacity_final1,
                    shs=shs_final1,
                    pipe=pipe,
                    bg_color=background,
                    cam_type=scene.dataset_type
                )
                _fwd_timings["render1"] += time() - _t
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg1["render"], render_pkg1["viewspace_points"], render_pkg1["visibility_filter"], render_pkg1["radii"]
                depth = render_pkg1["depth"].detach()

                _t = time()
                render_pkg2_1 = render_flow(
                    viewpoint_camera=viewpoint_cam2,
                    pc=gaussians,
                    xyz=means3D_final2_1,
                    scales=scales_final2_1,
                    rotations=rotations_final2_1,
                    opacity=opacity_final2_1,
                    shs=shs_final2_1,
                    pipe=pipe,
                    bg_color=background,
                    cam_type=scene.dataset_type
                )
                _fwd_timings["render2_1"] += time() - _t
                alpha, proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = render_pkg2_1[
                    "alpha"], render_pkg2_1["proj_2D"], render_pkg2_1["conic_2D"], render_pkg2_1["conic_2D_inv"
                    ], render_pkg2_1["gs_per_pixel"], render_pkg2_1["weight_per_gs_pixel"], render_pkg2_1["x_mu"]

                _t = time()
                render_pkg2 = render_flow(
                    viewpoint_camera=viewpoint_cam2,
                    pc=gaussians,
                    xyz=means3D_final2,
                    scales=scales_final2,
                    rotations=rotations_final2,
                    opacity=opacity_final2,
                    shs=shs_final2,
                    pipe=pipe,
                    bg_color=background,
                    cam_type=scene.dataset_type
                )
                _fwd_timings["render2"] += time() - _t
                next_proj_2D, next_conic_2D = render_pkg2["proj_2D"], render_pkg2["conic_2D"]
                if hyper.time_smoothness_weight != 0:
                    # tv_loss = 0
                    tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
                    loss += tv_loss

                _t = time()
                # warp gs_flow to match motion flow
                gs_flow = calculate_gs_flow(gs_per_pixel, weight_per_gs_pixel, next_conic_2D, conic_2D_inv, proj_2D, next_proj_2D, x_mu)
                gs_flow = warping_gs_flow(depth, gs_flow, viewpoint_cam1, viewpoint_cam2)
                _fwd_timings["gs_flow"] += time() - _t
            
            images.append(image.unsqueeze(0))

            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam1.original_image.cuda()
                next_gt_image = viewpoint_cam2.original_image.cuda()
            else:
                gt_image = viewpoint_cam1['image'].cuda()
                next_gt_image = viewpoint_cam2['image'].cuda()
                
            H, W = gt_image.shape[-2:]
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        t_forward_end = time()

        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
        loss += Ll1
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm

        t_flow_start = time()
        if stage == "fine":
            cache_key = (viewpoint_cam1.image_name, viewpoint_cam2.image_name)
            flow_cache_hit = cache_key in flow_cache
            if not flow_cache_hit:
                with torch.no_grad():
                    flow_pred = flownet(gt_image[None]*255, next_gt_image[None]*255)
                    H_flow, W_flow = flow_pred[0].shape[-2:]
                    if W_flow == W and H_flow == H:
                        flow_stored = flow_pred[0].squeeze()
                    else:
                        flow_stored = torch.nn.functional.interpolate(flow_pred[0], size=(H, W), mode="bilinear").squeeze()
                        flow_stored[0] *= W / W_flow
                        flow_stored[1] *= H / H_flow
                    flow_cache[cache_key] = flow_stored
            flow_2d_gt = flow_cache[cache_key]

            # TODO: add camera flow calculation, since cameras might not be the same.
            with torch.no_grad():
                camera_flow = calculate_camera_flow(depth, viewpoint_cam1, viewpoint_cam2)
                motion_flow = flow_2d_gt - camera_flow
                # motion_flow = motion_flow * (1 - motion_mask) if motion_mask is not None else motion_flow
            Lflow = flow_loss(gs_flow, motion_flow.detach(), H, W)
            loss += opt.flow_loss_weight * Lflow
        t_flow_end = time()

        t_backward_start = time()
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        t_backward_end = time()
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        t_optim_start = time()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_helper, [pipe, background], stage, scene.dataset_type)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                    # breakpoint()
                        render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            t_optim_end = time()
            t_iter_total = time() - t_iter_start

            # Per-step timing log
            if iteration % 100 == 0:
                t_data   = t_data_end    - t_data_start
                t_fwd    = t_forward_end - t_forward_start
                t_flow   = t_flow_end    - t_flow_start
                t_bwd    = t_backward_end - t_backward_start
                t_optim  = t_optim_end   - t_optim_start
                print(
                    f"[ITER {iteration}][{stage}] step={t_iter_total*1000:.1f}ms | "
                    f"data={t_data*1000:.1f}ms | fwd={t_fwd*1000:.1f}ms | "
                    f"flow={t_flow*1000:.1f}ms | bwd={t_bwd*1000:.1f}ms | "
                    f"optim={t_optim*1000:.1f}ms"
                    + (f" [flow cache miss]" if stage == "fine" and not flow_cache_hit else "")
                )
                if "fine" in stage:
                    print(
                        f"  fwd breakdown: "
                        f"deform1={_fwd_timings['deform1']*1000:.1f}ms | "
                        f"deform2={_fwd_timings['deform2']*1000:.1f}ms | "
                        f"deform_delta={_fwd_timings['deform_delta']*1000:.1f}ms | "
                        f"render1={_fwd_timings['render1']*1000:.1f}ms | "
                        f"render2_1={_fwd_timings['render2_1']*1000:.1f}ms | "
                        f"render2={_fwd_timings['render2']*1000:.1f}ms | "
                        f"gs_flow={_fwd_timings['gs_flow']*1000:.1f}ms"
                    )
                elif "coarse" in stage:
                    print(f"  fwd breakdown: render_coarse={_fwd_timings['render_coarse']*1000:.1f}ms")
            if tb_writer:
                tb_writer.add_scalar(f'{stage}/timing/step_ms',    (time() - t_iter_start) * 1000, iteration)
                tb_writer.add_scalar(f'{stage}/timing/data_ms',    (t_data_end - t_data_start) * 1000, iteration)
                tb_writer.add_scalar(f'{stage}/timing/forward_ms', (t_forward_end - t_forward_start) * 1000, iteration)
                tb_writer.add_scalar(f'{stage}/timing/flow_ms',    (t_flow_end - t_flow_start) * 1000, iteration)
                tb_writer.add_scalar(f'{stage}/timing/backward_ms',(t_backward_end - t_backward_start) * 1000, iteration)
                if "fine" in stage:
                    tb_writer.add_scalar(f'{stage}/timing/fwd_deform1_ms',      _fwd_timings["deform1"]      * 1000, iteration)
                    tb_writer.add_scalar(f'{stage}/timing/fwd_deform2_ms',      _fwd_timings["deform2"]      * 1000, iteration)
                    tb_writer.add_scalar(f'{stage}/timing/fwd_deform_delta_ms', _fwd_timings["deform_delta"] * 1000, iteration)
                    tb_writer.add_scalar(f'{stage}/timing/fwd_render1_ms',      _fwd_timings["render1"]      * 1000, iteration)
                    tb_writer.add_scalar(f'{stage}/timing/fwd_render2_1_ms',    _fwd_timings["render2_1"]    * 1000, iteration)
                    tb_writer.add_scalar(f'{stage}/timing/fwd_render2_ms',      _fwd_timings["render2"]      * 1000, iteration)
                    tb_writer.add_scalar(f'{stage}/timing/fwd_gs_flow_ms',      _fwd_timings["gs_flow"]      * 1000, iteration)
                elif "coarse" in stage:
                    tb_writer.add_scalar(f'{stage}/timing/fwd_render_coarse_ms', _fwd_timings["render_coarse"] * 1000, iteration)

            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()


            # Optimizer step
            t_optim_start = time()
            if iteration < opt.iterations:
                # Clip gradients to prevent NaN
                all_params = []
                for group in gaussians.optimizer.param_groups:
                    all_params.extend(group['params'])
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")
def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
