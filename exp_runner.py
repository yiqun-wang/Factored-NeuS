import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset, DatasetGlossySynthetic, DatasetGlossyReal, DatasetShiny, DatasetSk3d, DatasetSynthetic
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF, RefColor
from models.inverRender import EnvmapMaterialNetwork
from models.renderer import NeuSRenderer
import imageio
from PIL import Image
import open3d as o3d
import json
from evaluation.shiny_eval import evaluation_shinyblender


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, type='dtu', surface_weight=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case) 
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)  
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir_geo']  
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.surface_weight = surface_weight
        self.type = type
        if self.type == 'dtu':
            self.dataset = Dataset(self.conf['dataset'])
        elif self.type == 'sk3d':
            self.dataset = DatasetSk3d(self.conf['dataset'])
        elif self.type == 'indisg_synthetic':
            self.dataset = DatasetSynthetic(self.conf['dataset'])
        elif self.type =='indisg_shiny':
            self.dataset = DatasetShiny(self.conf['dataset'])
        elif self.type =='glossy_synthetic':
            self.dataset = DatasetGlossySynthetic(self.conf['dataset'])
        elif self.type =='glossy_real':
            self.dataset = DatasetGlossyReal(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')  # 512
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []  # 67
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)  # color
        # ref
        self.refColor_network = RefColor().to(self.device)

        params_to_train += list(self.nerf_outside.parameters())  # 24
        params_to_train += list(self.sdf_network.parameters())  # 27
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        # ref
        params_to_train += list(self.refColor_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(**self.conf['model.neus_renderer'],
                                     nerf=self.nerf_outside,
                                     sdf_network=self.sdf_network,
                                     deviation_network=self.deviation_network,
                                     color_network=self.color_network,
                                     refColor_network=self.refColor_network
                                    )

        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        if self.mode[:5] == 'train':
            self.file_backup()
            
        self.pre_train_iter = 0

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))  
        self.update_learning_rate()  
        res_step = self.end_iter - self.iter_step  
        image_perm = self.get_image_perm()  

        for iter_i in tqdm(range(res_step)):  
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)  # [batch,1]

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)  # [512,1]

            mask_sum = mask.sum() + 1e-5  # [512,]
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']  # batch, 3
            s_val = render_out['s_val']  # batch, 1
            cdf_fine = render_out['cdf_fine']  # batch, uniform+importance
            gradient_error = render_out['gradient_error']  # tensor
            weight_max = render_out['weight_max']  # batch, 1
            weight_sum = render_out['weight_sum']  # batch, 1

            surface_color = render_out['surface_color']
            sdf_mask = render_out['sdf_mask']
            mask_sdf_sum = mask[sdf_mask].sum() + 1e-5

            # Loss
            color_error = (color_fine - true_rgb) * mask  # batch, 3
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum  # tensor
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())  # tensor
            
            surface_color_error = self.surface_weight * (surface_color[sdf_mask] - true_rgb[sdf_mask]) * mask[sdf_mask]
            surface_color_loss = F.l1_loss(surface_color_error, torch.zeros_like(surface_color_error), reduction='sum') / mask_sdf_sum
            
            eikonal_loss = gradient_error  # tensor

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)  # tensor
            
            loss = color_fine_loss + \
                   surface_color_loss + \
                   eikonal_loss * self.igr_weight + \
                   mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
            

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:  
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:  
                if self.type == 'dtu' or self.type == 'sk3d' or self.type == 'glossy_synthetic' or self.type == 'glossy_real':  
                    self.validate_image()
                else:
                    self.validate_synthetic_img()

            if self.iter_step % self.val_mesh_freq == 0:  
                if self.type == 'dtu' or self.type == 'sk3d':
                    self.validate_mesh(world_space=True)
                elif self.type == 'shiny_refneus':
                    self.validate_mesh_shiny()
                else:
                    self.validate_mesh(world_space=False)

            self.update_learning_rate()  

            if self.iter_step % len(image_perm) == 0:  
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)  

    def get_cos_anneal_ratio(self):  
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:  
            learning_factor = self.iter_step / self.warm_up_end
        else:  
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor  

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:  # ['./', './models']
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':  
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.refColor_network.load_state_dict(checkpoint['refColor_network'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(), 
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'refColor_network': self.refColor_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))


    def validate_synthetic_img(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level  # 4

        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)  # H, W, 3
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)  # H*W/batch, batch, 3
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        
        out_diffuse = []
        out_specular = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('specular_color') and feasible('diffuse_color'):
                diffuse = render_out['diffuse_color'].detach().cpu().numpy()
                specular = render_out['specular_color'].detach().cpu().numpy()
                out_diffuse.append(diffuse)
                out_specular.append(specular)
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                # n_samples = self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        tonemap_img = lambda x: np.power(x, 1./2.2)

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3])
            img_fine = (tonemap_img(img_fine) * 255).clip(0, 255)

        diffuse_img = None
        if len(out_diffuse) > 0:
            diffuse_img = np.concatenate(out_diffuse, axis=0).reshape([H, W, 3])
            diffuse_img = (tonemap_img(diffuse_img) * 255).clip(0, 255)

        specular_img = None
        if len(out_specular) > 0:
            specular_img = np.concatenate(out_specular, axis=0).reshape([H, W, 3])
        
        normal_img = None
        if len(out_normal_fine) > 0:  
            normal_img = np.concatenate(out_normal_fine, axis=0)
            # rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            # normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([H, W, 3]) * 128 + 128).clip(0, 255)
            normal_img = (np.concatenate(out_normal_fine, axis=0).reshape([H, W, 3]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'diffuse'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'specular'), exist_ok=True)

        val_img = np.concatenate([img_fine, self.dataset.image_at(idx, resolution_level=resolution_level)])
        val_img = Image.fromarray(val_img.astype(np.uint8))
        val_img.save(os.path.join(self.base_exp_dir, 'validations_fine/v_{}_{}.png'.format(self.iter_step, idx)))
        
        diffuse_img = Image.fromarray(diffuse_img.astype(np.uint8))
        diffuse_img.save(os.path.join(self.base_exp_dir, 'diffuse/d_{}_{}.png'.format(self.iter_step, idx)))

        normal_img = Image.fromarray(normal_img.astype(np.uint8))
        normal_img.save(os.path.join(self.base_exp_dir, 'normals/n_{}_{}.png'.format(self.iter_step, idx)))
        
        cv.imwrite(os.path.join(self.base_exp_dir, 'specular/s_{}_{}.png'.format(self.iter_step, idx)), specular_img)
        # cv.imwrite(os.path.join(self.base_exp_dir, 'normals/n_{}_{}.png'.format(self.iter_step, idx)), normal_img)


    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level  # 4

        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)  # H, W, 3
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)  # H*W/batch, batch, 3
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        
        out_diffuse = []
        out_specular = []
        out_cd_plus_cs = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('specular_color') and feasible('diffuse_color'):
                diffuse = render_out['diffuse_color'].detach().cpu().numpy()
                specular = render_out['specular_color'].detach().cpu().numpy()
                out_diffuse.append(diffuse)
                out_specular.append(specular)

                surface_color = render_out['surface_color'].detach().cpu().numpy()
                out_cd_plus_cs.append(surface_color)
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                # n_samples = self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        diffuse_img = None
        if len(out_diffuse) > 0:
            diffuse_img = (np.concatenate(out_diffuse, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        specular_img = None
        if len(out_specular) > 0:
            specular_img = (np.concatenate(out_specular, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        cd_plus_cs_img = None
        if len(out_cd_plus_cs) > 0:
            cd_plus_cs_img = (np.concatenate(out_cd_plus_cs, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'diffuse'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'specular'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'CdPlusCs'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        'v_{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        'n_{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

            if len(out_diffuse) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'diffuse',
                                        'd_{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           diffuse_img[..., i])

            if len(out_specular) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'specular',
                                        's_{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           specular_img[..., i])
                
            if len(out_cd_plus_cs) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'CdPlusCs',
                                        'DPlusS_{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           cd_plus_cs_img[..., i])
            
                
    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine


    def validate_mesh(self, world_space=False, resolution=512, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:  # false
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')


    def validate_mesh_shiny(self, resolution=64, threshold=0.0, ckpt_path=None, validate_normal=False):
        result = open(os.path.join(self.base_exp_dir, 'result.txt'), 'a')  
        
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
        self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', 'inter_mesh.ply'))
        
        # For visibility identification
        mesh_ = o3d.io.read_triangle_mesh(os.path.join(self.base_exp_dir, 'meshes', 'inter_mesh.ply'))
        mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(mesh_)
        scene = o3d.t.geometry.RaycastingScene()
        cube_id = scene.add_triangles(mesh_)
        
        if self.iter_step % 10000 == 0 and self.iter_step != 0: 
            resolution = 512
            # 度量indisg时注释掉
            vertices, triangles =\
                self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
            os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
            
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

            mesh.apply_transform(self.dataset.scale_mat)  #transform to orignial space for evaluation
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_eval.ply'.format(self.iter_step)))
            mesh_eval = o3d.io.read_triangle_mesh(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_eval.ply'.format(self.iter_step)))
            # indisg
            # mesh_eval = o3d.io.read_triangle_mesh(os.path.join(self.base_exp_dir, 'meshes/mesh.ply'.format(self.iter_step)))
            # mesh = trimesh.Trimesh(mesh_eval.vertices, mesh_eval.triangles)
            # scale_mat = np.diag([300., 300., 300., 1.0]).astype(np.float32)
            # mesh.apply_transform(scale_mat)
            # mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_eval.ply'.format(self.iter_step)))
            # mesh_eval = o3d.io.read_triangle_mesh(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_eval.ply'.format(self.iter_step)))
            
            with open(os.path.join(self.conf['dataset'].data_dir, 'test_info.json'), 'r') as f:
                text_info = json.load(f)
            points_for_plane = text_info['points']
            max_dist_d = text_info['max_dist_d']
            max_dist_t = text_info['max_dist_t']
            try:
                nonvalid_bbox = text_info['nonvalid_bbox']
            except:
                nonvalid_bbox = None
            
            # refneus dense_pcd.ply
            mean_d2s, mean_s2d, over_all = evaluation_shinyblender(mesh_eval, os.path.join(self.conf['dataset'].data_dir, 'dense_pcd.ply'),self.base_exp_dir, 
                                                                   max_dist_d=max_dist_d, max_dist_t=max_dist_t, points_for_plane=points_for_plane, nonvalid_bbox=nonvalid_bbox )
            
            # indisg points_of_interest.ply scale_mat = [2,2,2,1]
            # mean_d2s, mean_s2d, over_all = evaluation_shinyblender(mesh_eval, os.path.join(self.conf['dataset'].data_dir, 'points_of_interest.ply'),self.base_exp_dir, 
            #                                                        max_dist_d=max_dist_d, max_dist_t=max_dist_t, points_for_plane=points_for_plane, nonvalid_bbox=nonvalid_bbox )

            result.write(f'{self.iter_step}: ')
            result.write(f'{mean_d2s} {mean_s2d} {over_all}')
            result.write('\n') 
            result.flush()
            # if self.iter_step == self.end_iter - 1 or ckpt_path is not None and validate_normal:
            #     self.validate_all_normals()

        logging.info('End')


    def mesh_dtu_shpere2world(self, mesh_name):
        mesh_sphere = o3d.io.read_triangle_mesh(os.path.join(self.base_exp_dir, 'meshes', f'{mesh_name}.ply'))
        vertices = mesh_sphere.vertices
        vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
        mesh = trimesh.Trimesh(vertices, mesh_sphere.triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '00300000.ply'))


    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                                                  resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()  
    parser.add_argument('--conf', type=str, default='./confs/base.conf')  
    parser.add_argument('--mode', type=str, default='train')  
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--type', type=str, default='dtu')
    parser.add_argument('--surface_weight', type=float, default=0.1)

    parser.add_argument('--idx', type=int, default=0)

    args = parser.parse_args() 

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)  

    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.type, args.surface_weight)  

    if args.mode == 'train':
        runner.train()  
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'validate_mesh_shiny':
        runner.validate_mesh_shiny()
    elif args.mode == 'mesh_dtu_shpere2world':
        runner.mesh_dtu_shpere2world(mesh_name='dtu122-300000')
    elif args.mode == 'validate_image':
        if args.type == 'dtu' or args.type == 'sk3d':
            runner.validate_image(resolution_level=1, idx=args.idx)
        elif args.type == 'synthetic':
            runner.validate_synthetic_img(idx=57, resolution_level=1)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
