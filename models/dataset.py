import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import json
from models import rend_util
import tifffile as tf
from copy import deepcopy
import trimesh


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]  # [3, 3]
    R = out[1]  # [3, 3]
    t = out[2]  # [4, 1]

    K = K / K[2, 2]
    intrinsics = np.eye(4)  
    intrinsics[:3, :3] = K  # [4,4]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]  # [4,4]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        # self.device = torch.device('cuda')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')  # cameras_sphere.npz
        self.object_cameras_name = conf.get_string('object_cameras_name')  # cameras_sphere.npz

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)  # n = 123
        # [n, h, w, c] or [123, 576, 768 ,3] normalized
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))  # n = 123
        # [n, h, w, c] or [123, 576, 768 ,3] normalized 
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # world_mat is a projection matrix from world to image
        # [n, 4, 4] world->image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []  # [n, 4, 4]

        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        '''
        f, 0, cx, 0
        0, f, cy, 0
        0, 0, 1 , 0
        0, 0, 0 , 1
        '''
        self.intrinsics_all = []  # [n, 4, 4] 
        self.pose_all = []  # [n, 4, 4] 

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat  
            P = P[:3, :4]  # [3, 4]
            intrinsics, pose = load_K_Rt_from_P(None, P)  # [4,4]     [4,4]
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]  # H: 576, W: 768
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        # [4,4]
        object_scale_mat = self.scale_mats_np[0]
        # [4,1] 
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        # [4,1]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        # [3,] [-1.01, -1.01000006, -1.01]
        self.object_bbox_min = object_bbox_min[:3, 0]
        # [3,] [1.01, 1.00999994, 1.01]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level  # 4
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)  # [w//l, h//l]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # [W, H, 3]
        # image->camera
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        # camera->camera unit sphere
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        # camera unit sphere->world unit sphere
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)  # H, W, 3

    def gen_random_rays_at(self, img_idx, batch_size):
        img_idx = img_idx.cpu()
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]).cpu()  # [512,]
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size]).cpu()  # [512,]
        color = self.images[img_idx][(pixels_y, pixels_x)]  # [batch_size, 3]
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # [batch_size, 3]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).cuda().float()  # [batch_size, 3]
        # image->camera [1, 3, 3]@[batch, 3, 1]=[batch, 3, 1]
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # [batch_size, 3]
        # camera->camera(unit sphere)
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # [batch_size, 3]
        # camera(unit sphere)->world(unit sphere)
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # [batch_size, 3]
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # [batch_size, 3]

        if torch.cuda.is_available():
            return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()  # [batch_size, 10]
        else:
            return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1)  # [batch_size, 10]

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)  # [batch,1]
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)  # [batch,1]
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
    

class DatasetSk3d:
    def __init__(self, conf):
        super(DatasetSk3d, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = 'tis_right/idr_input/cameras.npz'
        self.object_cameras_name = 'tis_right/idr_input/cameras.npz'

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'tis_right/rgb/undistorted/ambient@best/*.png')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        
        # self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        # self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        # self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        self.roi_boxes = [camera_dict['roi_box_%d' % idx] for idx in range(self.n_images)] 
        self.sample_roi_prob = conf.get_float('sample_roi_prob', default=0.0)
        assert 0.0 <= self.sample_roi_prob <= 1.0

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        device = self.images.device

        if self.sample_roi_prob == 0.0:
            pixels_x = torch.randint(low=0, high=self.W, size=[batch_size], device=device)
            pixels_y = torch.randint(low=0, high=self.H, size=[batch_size], device=device)
        else:
            left, right, top, bottom = self.roi_boxes[img_idx]
            left, right = max(0, left-10), min(self.W, right+10)
            top, bottom = max(0, top-10), min(self.H, bottom+10)

            # sample rays inside the region of interest
            in_n = int(batch_size * self.sample_roi_prob)
            in_x = torch.randint(low=left, high=right, size=[in_n], device=device)
            in_y = torch.randint(low=top, high=bottom, size=[in_n], device=device)
            
            # sample rays outside the (square donut-like) region of interest 
            out_n = batch_size - in_n
            xx = torch.arange(0, self.W, device=device).view(1, self.W).expand(self.H, self.W)
            yy = torch.arange(0, self.H, device=device).view(self.H, 1).expand(self.H, self.W)
            xy = torch.stack([xx, yy], dim=0)
            xy[:, top:bottom, left:right] = -1
            out_x = torch.masked_select(xy[0], xy[0]>=0)
            out_y = torch.masked_select(xy[1], xy[1]>=0)
            indices = torch.randint(low=0, high=len(out_x), size=[out_n], device=out_x.device)
            out_x, out_y = out_x[indices], out_y[indices]

            # concatenate rays
            pixels_x = torch.cat([in_x, out_x])
            pixels_y = torch.cat([in_y, out_y])
        
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        # mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        mask = ((255.0 / 256.0) * color.new_ones(1, 1)).expand(color.shape)  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = p.to(self.intrinsics_all_inv.device)
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


class DatasetSynthetic:
    def __init__(self, 
                 conf,
                 frame_skip=1,
                 split='train'  
                 ):
        print('Load data: Begin')
        # self.device = torch.device('cuda')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf
        self.split = split
        # self.split = 'test'  # validate image 

        self.data_dir = conf.get_string('data_dir')

        # transform
        json_path = os.path.join(self.data_dir, 'transforms_{}.json'.format(split))
        with open(json_path, 'r') as fp:
            meta = json.load(fp)

        image_paths = []
        mask_paths = []
        rough_paths = []
        albedo_paths = []
        poses = []
        for frame in meta['frames']:
            poses.append(np.array(frame['transform_matrix']))
            if split == 'train':
                image_paths.append(os.path.join(self.data_dir, frame['file_path'] + '_rgb.exr'))
                mask_paths.append(os.path.join(self.data_dir, frame['file_path'] + '_mask.png'))
            if split == 'test':
                image_paths.append(os.path.join(self.data_dir, frame['file_path'] + '_rgba.png'))
                rough_paths.append(os.path.join(self.data_dir, frame['file_path'] + '_rough.png'))
                albedo_paths.append(os.path.join(self.data_dir, frame['file_path'] + '_albedo.png'))

        img_h, img_w = rend_util.load_rgb(image_paths[0]).shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * img_w / np.tan(.5 * camera_angle_x)
        poses = np.array(poses)
        scale = 2.0
        poses[..., 3] /= scale 

        # skip for training
        image_paths = image_paths[::frame_skip]
        poses = poses[::frame_skip, ...]
        self.image_paths = image_paths

        self.intrinsics_all = []  # [n_images, 3, 3]
        self.pose_all = []  # [n_images, 4, 4]
        intrinsics = [[focal, 0, img_w / 2],[0, focal, img_h / 2], [0, 0, 1]]
        intrinsics = np.array(intrinsics).astype(np.float32)
        for i in range(len(image_paths)):
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(poses[i]).float())

        self.rgb_images = []  # [n_images, h, w, 3]
        self.object_masks = []  # [n_images, h, w]
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            self.rgb_images.append(torch.from_numpy(rgb).float())
        
        if split == 'train':
            for path in mask_paths:
                object_mask = rend_util.load_mask(path)
                self.object_masks.append(torch.from_numpy(object_mask))
        
        self.rough = []  # n,h,w,3
        self.albedo = []  # n,h,w,3
        if split == 'test':
            for path in rough_paths:
                rough = rend_util.load_rgb(path)
                self.rough.append(torch.from_numpy(rough).float())
            for path in albedo_paths:
                albedo = rend_util.load_rgb(path)
                self.albedo.append(torch.from_numpy(albedo).float())
            self.rough = torch.stack(self.rough)
            self.albedo = torch.stack(self.albedo)

        
        convert_mat = torch.zeros([4, 4])
        convert_mat[0, 0] = 1.0
        convert_mat[1, 1] = -1.0
        convert_mat[2, 2] = -1.0
        convert_mat[3, 3] = 1.0
        
        self.n_images = len(image_paths)
        self.images = torch.stack(self.rgb_images).cpu()  # [n_images, h, w, 3]
        self.images_lis = image_paths
        if split == 'train': 
            self.masks = torch.stack(self.object_masks, dim=0).cpu().reshape(self.n_images, img_h, img_w, 1).repeat(1,1,1,3)  # [n_images, h, w, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all, dim=0).to(self.device)  # [n_images, 3, 3]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.focal = focal
        self.pose_all = torch.stack(self.pose_all).to(self.device)  @ convert_mat  # [n_images, 4, 4]
        self.H,self.W = self.images.shape[1], self.images.shape[2]
        self.images_pixels = self.H * self.W

        self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.array([1.01, 1.01, 1.01])

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """

        l = resolution_level  # 4
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)  # [w//l, h//l]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # [W, H, 3]
        # image->camera
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        # camera->camera unit sphere
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        # camera unit sphere->world unit sphere
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)  # H, W, 3

    def gen_random_rays_at(self, img_idx, batch_size):
        img_idx = img_idx.cpu()
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]).cpu()  # [512,]
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size]).cpu()  # [512,]
        color = self.images[img_idx][(pixels_y, pixels_x)]  # [batch_size, 3]
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # [batch_size, 3]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).cuda().float()  # [batch_size, 3]
        # image->camera [1, 3, 3]@[batch, 3, 1]=[batch, 3, 1]
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # [batch_size, 3]
        # camera->camera(unit sphere)
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # [batch_size, 3]
        # camera(unit sphere)->world(unit sphere)
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # [batch_size, 3]
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # [batch_size, 3]

        if torch.cuda.is_available():
            return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()  # [batch_size, 10]
        else:
            return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1)  # [batch_size, 10]

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)  # [batch,1]
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)  # [batch,1]
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):  
        img = np.power(rend_util.load_rgb(self.images_lis[idx]), 1./2.2) * 255  # exr->npg [h,w,3]
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


class DatasetShiny:  
    def __init__(self, 
                 conf,
                 frame_skip=1,
                 split='train'  
                 ):
        print('Load data: Begin')
        # self.device = torch.device('cuda')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf

        # split = 'test'  # validate
        self.split = split

        self.data_dir = conf.get_string('data_dir')

        # transform
        json_path = os.path.join(self.data_dir, 'transforms_{}.json'.format(split))
        with open(json_path, 'r') as fp:
            meta = json.load(fp)

        image_paths = []
        mask_paths = []
        poses = []
        for frame in meta['frames']:
            poses.append(np.array(frame['transform_matrix']))
            image_paths.append(os.path.join(self.data_dir, frame['file_path'] + '.png'))  # rgba.png
            if 'ball' not in self.data_dir:
                mask_paths.append(os.path.join(self.data_dir, frame['file_path'] + '_disp.tiff'))  
            else:
                mask_paths.append(os.path.join(self.data_dir, frame['file_path'] + '_alpha.png'))

        img_h, img_w = rend_util.load_rgb(image_paths[0]).shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * img_w / np.tan(.5 * camera_angle_x)
        poses = np.array(poses)
        scale = 2.0
        poses[..., 3] /= scale  # 

        # self.scale_mat = np.diag([300., 300., 300., 1.0]).astype(np.float32)

        # skip for training
        image_paths = image_paths[::frame_skip]
        poses = poses[::frame_skip, ...]
        self.image_paths = image_paths

        self.intrinsics_all = []  # [n_images, 3, 3]
        self.pose_all = []  # [n_images, 4, 4]
        intrinsics = [[focal, 0, img_w / 2],[0, focal, img_h / 2], [0, 0, 1]]
        intrinsics = np.array(intrinsics).astype(np.float32)
        for i in range(len(image_paths)):
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(poses[i]).float())

        self.rgb_images = []  # [n_images, h, w, 3]
        self.object_masks = []  # [n_images, h, w]
        for path in image_paths:
            rgb = rend_util.load_rgb(path)  
            self.rgb_images.append(torch.from_numpy(rgb).float())

        for path in mask_paths:
            if 'ball' not in self.data_dir:
                disp = tf.imread(path)  # [h,w]
                disp[disp > 1e-6] = 1.
            else:
                disp = cv.imread(path) / 256.
                disp[disp > 0.5] = 1.  # h,w,3
                disp = np.mean(disp, axis=-1, keepdims=False)  # h,w
            self.object_masks.append(torch.from_numpy(np.float32(disp)))            

        
        convert_mat = torch.zeros([4, 4])
        convert_mat[0, 0] = 1.0
        convert_mat[1, 1] = -1.0
        convert_mat[2, 2] = -1.0
        convert_mat[3, 3] = 1.0
        
        self.n_images = len(image_paths)
        self.images = torch.stack(self.rgb_images).cpu()  # [n_images, h, w, 3]
        self.images_lis = image_paths
        self.masks = torch.stack(self.object_masks, dim=0).cpu().reshape(self.n_images, img_h, img_w, 1).repeat(1,1,1,3)  # [n_images, h, w, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all, dim=0).to(self.device)  # [n_images, 3, 3]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.focal = focal
        self.pose_all = torch.stack(self.pose_all).to(self.device)  @ convert_mat  # [n_images, 4, 4]
        self.H,self.W = self.images.shape[1], self.images.shape[2]
        self.images_pixels = self.H * self.W

        self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.array([1.01, 1.01, 1.01])

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """

        l = resolution_level  # 4
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)  # [w//l, h//l]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # [W, H, 3]
        # image->camera
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        # camera->camera unit sphere
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        # camera unit sphere->world unit sphere
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)  # H, W, 3

    def gen_random_rays_at(self, img_idx, batch_size):
        img_idx = img_idx.cpu()
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]).cpu()  # [512,]
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size]).cpu()  # [512,]
        color = self.images[img_idx][(pixels_y, pixels_x)]  # [batch_size, 3]
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # [batch_size, 3]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).cuda().float()  # [batch_size, 3]
        # image->camera [1, 3, 3]@[batch, 3, 1]=[batch, 3, 1]
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # [batch_size, 3]
        # camera->camera(unit sphere)
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # [batch_size, 3]
        # camera(unit sphere)->world(unit sphere)
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # [batch_size, 3]
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # [batch_size, 3]

        if torch.cuda.is_available():
            return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()  # [batch_size, 10]
        else:
            return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1)  # [batch_size, 10]

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)  # [batch,1]
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)  # [batch,1]
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):  
        img = np.power(rend_util.load_rgb(self.images_lis[idx]), 1./2.2) * 255  # exr->npg [h,w,3]
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


import pickle
def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
    


class DatasetGlossySynthetic:
    def __init__(self, 
                 conf,
                 frame_skip=1,
                 ):
        print('Load data: Begin')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')

        self.img_num = len(glob(f'{self.data_dir}/*.pkl'))
        self.img_ids= [str(k) for k in range(self.img_num)]
        self.cams = [read_pickle(f'{self.data_dir}/{k}-camera.pkl') for k in range(self.img_num)]
        self.scale_factor = 1.

        self.images_lis = []
        self.masks_lis = []
        self.intrinsics_all = []
        self.pose_all = []
        for img_id in self.img_ids:
            # img
            img = cv.imread(f'{self.data_dir}/{img_id}.png')[...,:3] / 256.0
            self.images_lis.append(img)
            # mask
            depth = cv.imread(f'{self.data_dir}/{img_id}-depth.png')[...,:3] / 256.0
            mask = depth < 0.9
            # mask = cv.imread(f'{self.data_dir}/colmap/masks/{img_id}.png')[...,:3] / 256.0
            self.masks_lis.append(mask)
            # K
            K = self.cams[int(img_id)][1]
            self.intrinsics_all.append(K)
            # pose
            pose = self.cams[int(img_id)][0].copy()
            pose[:,3:] *= self.scale_factor
            self.pose_all.append(pose)

        self.images_np = np.stack(self.images_lis)
        self.masks_np = np.stack(self.masks_lis)
        self.intrinsics_all_np = np.stack(self.intrinsics_all).astype(np.float32)
        self.pose_all_np = np.stack(self.pose_all).astype(np.float32)

        self.n_images = len(self.images_lis)

        # convert_mat = torch.zeros([4, 4])
        # convert_mat[0, 0] = 1.0
        # convert_mat[1, 1] = -1.0
        # convert_mat[2, 2] = -1.0
        # convert_mat[3, 3] = 1.0
        convert_mat = torch.eye(4)

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()
        self.intrinsics_all = torch.from_numpy(self.intrinsics_all_np).to(self.device)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.from_numpy(self.pose_all_np).to(self.device) @ convert_mat  # [3, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.array([1.01, 1.01, 1.01])

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level  # 4
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)  # [w//l, h//l]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # [W, H, 3]
        
        # nero ####################################################################################################################

        rays_v = p @ self.intrinsics_all_inv[img_idx, :3, :3].permute(1, 0)  # W, H, 3
        rays_v = self.pose_all[img_idx, None, None, :, :3].permute(0, 1, 3, 2) @ rays_v[..., None]  # W, H, 3, 1
        rays_v = rays_v[..., 0]
        rays_v = rays_v / torch.linalg.norm(rays_v, ord=2, dim=-1, keepdim=True)  # W, H, 3

        rays_o = self.pose_all[:, :, :3].permute(0, 2, 1) @ -self.pose_all[:, :, 3:]
        rays_o = rays_o[img_idx, None, None, :, 0].expand(rays_v.shape)

        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)  # H, W, 3

    def gen_random_rays_at(self, img_idx, batch_size):
        img_idx = img_idx.cpu()
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]).cpu()  # [512,]
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size]).cpu()  # [512,]
        color = self.images[img_idx][(pixels_y, pixels_x)]  # [batch_size, 3]
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # [batch_size, 3]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).cuda().float()  # [batch_size, 3]

        # nero ####################################################################################################################

        rays_v = p @ self.intrinsics_all_inv[img_idx, :3, :3].permute(1, 0)
        rays_v = self.pose_all[img_idx, None, :, :3].expand(rays_v.shape[0], 3, 3).permute(0, 2, 1) @ rays_v[..., None]
        rays_v = rays_v[..., 0]
        rays_v = rays_v / torch.linalg.norm(rays_v, ord=2, dim=-1, keepdim=True)  # [batch_size, 3]

        rays_o = self.pose_all[:, :, :3].permute(0, 2, 1) @ -self.pose_all[:, :, 3:]
        rays_o = rays_o[img_idx, None, :, 0].expand(rays_v.shape)

        if torch.cuda.is_available():
            return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()  # [batch_size, 10]
        else:
            return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1)  # [batch_size, 10]

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)  # [batch,1]
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)  # [batch,1]
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):  
        img = self.images_lis[idx] * 256  
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


from colmap import plyfile
from pathlib import Path
from tqdm import tqdm


class DatasetGlossyReal:
    def __init__(self, 
                 conf,
                 frame_skip=1,
                 ):
        print('Load data: Begin')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf

        self.meta_info={
            'bear': {'forward': np.asarray([0.539944,-0.342791,0.341446],np.float32), 'up': np.asarray((0.0512875,-0.645326,-0.762183),np.float32),},
            'coral': {'forward': np.asarray([0.004226,-0.235523,0.267582],np.float32), 'up': np.asarray((0.0477973,-0.748313,-0.661622),np.float32),},
            'maneki': {'forward': np.asarray([-2.336584, -0.406351, 0.482029], np.float32), 'up': np.asarray((-0.0117387, -0.738751, -0.673876), np.float32), },
            'bunny': {'forward': np.asarray([0.437076,-1.672467,1.436961],np.float32), 'up': np.asarray((-0.0693234,-0.644819,-.761185),np.float32),},
            'vase': {'forward': np.asarray([-0.911907, -0.132777, 0.180063], np.float32), 'up': np.asarray((-0.01911, -0.738918, -0.673524), np.float32), },
        }

        self.data_dir = conf.get_string('data_dir')
        self.object_name = self.data_dir.split('/')[-2]
        self.poses, self.Ks, self.image_names, self.img_ids = read_pickle(f'{self.data_dir}/cache.pkl')
        self._normalize()

        h, w, _ = cv.imread(f'{self.data_dir}/images/{self.image_names[self.img_ids[0]]}')[...,:3].shape
        self.max_len = 1024
        ratio = float(self.max_len) / max(h, w)
        th, tw = int(ratio*h), int(ratio*w)
        rh, rw = th / h, tw / w

        self.images_lis = []
        self.intrinsics_all = []
        self.pose_all = []
        for img_id in self.img_ids:
            # img
            img = cv.imread(f'{self.data_dir}/images_raw_1024/{self.image_names[img_id]}')[...,:3] / 256.0
            self.images_lis.append(img)
            # K
            K = self.Ks[img_id]
            K = np.diag([rw,rh,1.0]) @ K
            self.intrinsics_all.append(K)
            # pose
            pose = self.poses[img_id]
            self.pose_all.append(pose)

        self.images_np = np.stack(self.images_lis)
        self.intrinsics_all_np = np.stack(self.intrinsics_all).astype(np.float32)
        self.pose_all_np = np.stack(self.pose_all).astype(np.float32)

        self.n_images = len(self.images_lis)

        # convert_mat = torch.zeros([4, 4])
        # convert_mat[0, 0] = 1.0
        # convert_mat[1, 1] = -1.0
        # convert_mat[2, 2] = -1.0
        # convert_mat[3, 3] = 1.0
        convert_mat = torch.eye(4)

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()
        self.masks = torch.ones_like(self.images).cpu()
        self.intrinsics_all = torch.from_numpy(self.intrinsics_all_np).to(self.device)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.from_numpy(self.pose_all_np).to(self.device) @ convert_mat  # [3, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.array([1.01, 1.01, 1.01])

    def _load_point_cloud(self, pcl_path):
        with open(pcl_path, "rb") as f:
            plydata = plyfile.PlyData.read(f)
            xyz = np.stack([np.array(plydata["vertex"][c]).astype(float) for c in ("x", "y", "z")], axis=1)
        return xyz
    
    def _compute_rotation(self, vert, forward):
        y = np.cross(vert, forward)
        x = np.cross(y, vert)

        vert = vert/np.linalg.norm(vert)
        x = x/np.linalg.norm(x)
        y = y/np.linalg.norm(y)
        R = np.stack([x, y, vert], 0)
        return R
    
    def _normalize(self):
        ref_points = self._load_point_cloud(f'{self.data_dir}/object_point_cloud.ply')
        max_pt, min_pt = np.max(ref_points, 0), np.min(ref_points, 0)
        center = (max_pt + min_pt) * 0.5
        offset = -center # x1 = x0 + offset
        scale = 1 / np.max(np.linalg.norm(ref_points - center[None,:], 2, 1)) # x2 = scale * x1
        up, forward = self.meta_info[self.object_name]['up'], self.meta_info[self.object_name]['forward']
        up, forward = up/np.linalg.norm(up), forward/np.linalg.norm(forward)
        R_rec = self._compute_rotation(up, forward) # x3 = R_rec @ x2
        self.ref_points = scale * (ref_points + offset) @ R_rec.T
        self.scale_rect = scale
        self.offset_rect = offset
        self.R_rect = R_rec

        # x3 = R_rec @ (scale * (x0 + offset))
        # R_rec.T @ x3 / scale - offset = x0

        # pose [R,t] x_c = R @ x0 + t
        # pose [R,t] x_c = R @ (R_rec.T @ x3 / scale - offset) + t
        # x_c = R @ R_rec.T @ x3 + (t - R @ offset) * scale
        # R_new = R @ R_rec.T    t_new = (t - R @ offset) * scale
        for img_id, pose in self.poses.items():
            R, t = pose[:,:3], pose[:,3]
            R_new = R @ R_rec.T
            t_new = (t - R @ offset) * scale
            self.poses[img_id] = np.concatenate([R_new, t_new[:,None]], -1)

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level  # 4
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)  # [w//l, h//l]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # [W, H, 3]
        
        # nero ####################################################################################################################

        rays_v = p @ self.intrinsics_all_inv[img_idx, :3, :3].permute(1, 0)  # W, H, 3
        rays_v = self.pose_all[img_idx, None, None, :, :3].permute(0, 1, 3, 2) @ rays_v[..., None]  # W, H, 3, 1
        rays_v = rays_v[..., 0]
        rays_v = rays_v / torch.linalg.norm(rays_v, ord=2, dim=-1, keepdim=True)  # W, H, 3

        rays_o = self.pose_all[:, :, :3].permute(0, 2, 1) @ -self.pose_all[:, :, 3:]
        rays_o = rays_o[img_idx, None, None, :, 0].expand(rays_v.shape)

        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)  # H, W, 3

    def gen_random_rays_at(self, img_idx, batch_size):
        img_idx = img_idx.cpu()
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]).cpu()  # [512,]
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size]).cpu()  # [512,]
        color = self.images[img_idx][(pixels_y, pixels_x)]  # [batch_size, 3]
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # [batch_size, 3]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).cuda().float()  # [batch_size, 3]

        # nero ####################################################################################################################

        rays_v = p @ self.intrinsics_all_inv[img_idx, :3, :3].permute(1, 0)
        rays_v = self.pose_all[img_idx, None, :, :3].expand(rays_v.shape[0], 3, 3).permute(0, 2, 1) @ rays_v[..., None]
        rays_v = rays_v[..., 0]
        rays_v = rays_v / torch.linalg.norm(rays_v, ord=2, dim=-1, keepdim=True)  # [batch_size, 3]

        rays_o = self.pose_all[:, :, :3].permute(0, 2, 1) @ -self.pose_all[:, :, 3:]
        rays_o = rays_o[img_idx, None, :, 0].expand(rays_v.shape)

        if torch.cuda.is_available():
            return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()  # [batch_size, 10]
        else:
            return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1)  # [batch_size, 10]

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)  # [batch,1]
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)  # [batch,1]
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):  
        img = self.images_lis[idx] * 256  
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
