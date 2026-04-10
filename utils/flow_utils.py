import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np 

from utils.loss_utils import l1_loss


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        # Points: B 4 HW 
        # K：B 4 4
        # T: B 4 4
        P = torch.matmul(K, T)[:, :3, :]  # B 3 4
        cam_points = torch.matmul(P, points)  # B 4 HW 
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2:3, :] + self.eps)  # B 2 HW
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width) # B 2 H W
        pix_coords = pix_coords.permute(0, 2, 3, 1) # B H W 2
        # normalize
        _pix_coords_ = torch.clone(pix_coords)
        _pix_coords_[..., 0] /= self.width - 1
        _pix_coords_[..., 1] /= self.height - 1
        _pix_coords_ = (_pix_coords_ - 0.5) * 2
        return _pix_coords_, pix_coords


def calculate_camera_flow(depth1, cam1, cam2):
    H, W = depth1.shape[-2:] # depth1: (B) (1) H W
    backprojdepth = BackprojectDepth(1, H, W).cuda()
    project3d = Project3D(1, H, W).cuda()
    inv_K1 = torch.linalg.inv(cam1.intrinsic.cuda())[None] # B 4 4
    K2 = cam2.intrinsic.cuda()[None] # B 4 4
    T12 = torch.matmul(torch.linalg.inv(cam2.extrinsic.cuda()), 
                     cam1.extrinsic.cuda())[None] # B 4 4
    points_3d = backprojdepth(depth1, inv_K1) # B 4 HW
    _, pixel_coords = project3d(points_3d, K2, T12) # B H W 2
    pixel_coords = pixel_coords.permute(0, 3, 1, 2) # B 2 H W
    ori_coords = backprojdepth.pix_coords.view(1, 3, H, W)[:, :2] # B 2 H W
    camere_flow = pixel_coords - ori_coords # B 2 H W
    return camere_flow[0] # 2 H W


def warping_gs_flow(depth_gt, gs_flow, camera_pose, next_camera_pose):
    H, W = depth_gt.shape[-2:] # depth1: (B) (1) H W
    backprojdepth = BackprojectDepth(1, H, W).cuda()
    project3d = Project3D(1, H, W).cuda()
    inv_K1 = torch.linalg.inv(camera_pose.intrinsic.cuda())[None] # B 4 4
    K2 = next_camera_pose.intrinsic.cuda()[None] # B 4 4
    T12 = torch.matmul(torch.linalg.inv(next_camera_pose.extrinsic.cuda()), 
                     camera_pose.extrinsic.cuda())[None] # B 4 4
    points_3d = backprojdepth(depth_gt, inv_K1) # B 4 HW
    pixel_coords_norm, _ = project3d(points_3d, K2, T12) # B H W 2
    gs_flow= F.grid_sample(gs_flow.unsqueeze(0), pixel_coords_norm, padding_mode="border", align_corners=True) # B 3 H W
    return gs_flow.squeeze(0)


def calculate_gs_flow(gs_per_pixel, weight_per_gs_pixel, next_conic_2D, conic_2D_inv, proj_2D, next_proj_2D, x_mu):
    # gs_per_pixel = gs_per_pixel.long() # K H W
    # # deal with empty gs
    # valid_mask = ~(gs_per_pixel < 0).any(dim=0) # H W
    # proj_2D_per_pixel = proj_2D[gs_per_pixel].permute(0, 3, 1, 2) # K 2 H W
    # next_proj_2D_per_pixel = next_proj_2D[gs_per_pixel].permute(0, 3, 1, 2) # K 2 H W
    # next_conic_2D_per_pixel = conic_to_matrix(next_conic_2D[gs_per_pixel].permute(0, 3, 1, 2)) # K 3 H W -> K 2 2 H W
    # conic_2D_inv_per_pixel = conic_to_matrix(conic_2D_inv[gs_per_pixel].permute(0, 3, 1, 2)) # K 3 H W -> K 2 2 H W
    # flow_per_pixel = torch.einsum("kabhw, kbchw, kchw -> kahw", [next_conic_2D_per_pixel, conic_2D_inv_per_pixel, 
    #                     x_mu]) + next_proj_2D_per_pixel - (x_mu + proj_2D_per_pixel) # K 2 H W
    # weight_per_gs_pixel = weight_per_gs_pixel / (weight_per_gs_pixel.sum(dim=0, keepdim=True) + 1e-7) # K H W
    # flow_gs = torch.einsum("khw, kahw -> ahw", [weight_per_gs_pixel, flow_per_pixel]) # 2 H W
    # return flow_gs * valid_mask.float()


    conic_2D_inv = conic_2D_inv.detach() # K 3

    gs_per_pixel = gs_per_pixel.long() # K H W
    # valid_mask = ~(gs_per_pixel < 0).any(dim=0) # H W
    conv_conv = torch.zeros([conic_2D_inv.shape[0], 2, 2], device=conic_2D_inv.device) # K 2 2
    # MATMUL
    conv_conv[:, 0, 0] = next_conic_2D[:, 0] * conic_2D_inv[:, 0] + next_conic_2D[:, 1] * conic_2D_inv[:, 1]
    conv_conv[:, 0, 1] = next_conic_2D[:, 0] * conic_2D_inv[:, 1] + next_conic_2D[:, 1] * conic_2D_inv[:, 2]
    conv_conv[:, 1, 0] = next_conic_2D[:, 1] * conic_2D_inv[:, 0] + next_conic_2D[:, 2] * conic_2D_inv[:, 1]
    conv_conv[:, 1, 1] = next_conic_2D[:, 1] * conic_2D_inv[:, 1] + next_conic_2D[:, 2] * conic_2D_inv[:, 2]

    # isotropic gs flow
    # flow_per_pixel = next_proj_2D[gs_per_pixel] - proj_2D[gs_per_pixel].detach() # K H W 3

    # anisotropic gs flow
    conv_multi = (conv_conv[gs_per_pixel] @ x_mu.permute(0,2,3,1).unsqueeze(-1).detach()).squeeze() # K H W 2
    flow_per_pixel = (conv_multi + next_proj_2D[gs_per_pixel] - proj_2D[gs_per_pixel].detach() - x_mu.permute(0,2,3,1).detach()) # K H W 2

    weight_per_gs_pixel = weight_per_gs_pixel / (weight_per_gs_pixel.sum(dim=0, keepdim=True) + 1e-7) # K H W
    flow_gs = torch.einsum("khw, khwa -> ahw", [weight_per_gs_pixel.detach(), flow_per_pixel]) # 2 H W
    return flow_gs


def flow_loss(flow_pred, flow_gt, height, width):
    flow_pred[0] /= height
    flow_pred[1] /= width
    flow_pred = flow_pred.clamp(-1, 1)
    flow_gt[0] /= height
    flow_gt[1] /= width
    flow_gt = flow_gt.clamp(-1, 1)
    return l1_loss(flow_pred, flow_gt)
