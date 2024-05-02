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
from models.dataset import Dataset, DatasetShiny, DatasetSk3d, DatasetSynthetic
from models.fields import IndirectLight, Lvis, RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF, RefColor
from models.inverRender import EnvmapMaterialNetwork
from models.renderer import NeuSRenderer
import imageio
from PIL import Image
from torch import autograd
from models import math_utils as utils


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, type='dtu'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)  
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text) 
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        
        self.base_exp_dir_mateIllu = self.conf['general.base_exp_dir_mateIllu'] 
        self.base_exp_dir_lvis = self.conf['general.base_exp_dir_lvis']
        os.makedirs(self.base_exp_dir_mateIllu, exist_ok=True)

        self.type = type
        if self.type == 'dtu':
            self.dataset = Dataset(self.conf['dataset'])
        elif self.type == 'sk3d':
            self.dataset = DatasetSk3d(self.conf['dataset'])
        elif self.type == 'synthetic':
            self.dataset = DatasetSynthetic(self.conf['dataset'])
        elif self.type =='shiny':
            self.dataset = DatasetShiny(self.conf['dataset'])
            
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.metaIllu.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        # self.val_mesh_freq = self.conf.get_int('train.metaIllu.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.metaIllu.batch_size')  # 512
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')  
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')  # 0.05
        
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

        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.refColor_network = RefColor().to(self.device)
        self.lvis_network = Lvis().to(self.device)
        self.indiLgt_network = IndirectLight().to(self.device)

        self.mateIllu_network = EnvmapMaterialNetwork().to(self.device)

        params_to_train += list(self.mateIllu_network.parameters())
        # params_to_train += list(self.sdf_network.parameters())  

        # self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad is not False, params_to_train), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        model_list_raw = os.listdir(os.path.join(self.base_exp_dir_lvis, 'checkpoints'))
        model_list = []
        for model_name in model_list_raw:
            if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.conf.get_int('train.lvis.end_iter'):
                model_list.append(model_name)
        model_list.sort()
        latest_model_name = model_list[-1]
        logging.info('Find lvis checkpoint: {}'.format(latest_model_name))
        self.load_checkpoint_lvis(latest_model_name)

        self.renderer = NeuSRenderer(**self.conf['model.neus_renderer'],
                                     sdf_network=self.sdf_network,
                                     deviation_network=self.deviation_network,
                                     refColor_network=self.refColor_network,
                                     lvis_network=self.lvis_network,
                                     indiLgt_network=self.indiLgt_network,
                                     mateIllu_network=self.mateIllu_network
                                     )

        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir_mateIllu, 'checkpoints'))
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
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir_mateIllu, 'logs'))  
        self.update_learning_rate() 
        res_step = self.end_iter - self.iter_step  
        image_perm = self.get_image_perm()  

        for iter_i in tqdm(range(res_step)):  
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)  # [batch,1]

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)  # [512,1]

            mask_sum = mask.sum() + 1e-5  # [512,]
            
            # with autograd.detect_anomaly():

            mateIllu_out = self.renderer.mateIllu_render(rays_o, rays_d, near, far)

            rgb = mateIllu_out['rgb']

            sdf_mask = mateIllu_out['sdf_mask']
            if sdf_mask.sum() < 1e-6: continue
            sdf_mask_sum = mask[sdf_mask].sum() + 1e-5

            encoder_loss = mateIllu_out['encoder_loss']
            smooth_loss = mateIllu_out['smooth_loss']
            
            rgb_error = (rgb[sdf_mask] - true_rgb[sdf_mask]) * mask[sdf_mask]
            rgb_loss = F.l1_loss(rgb_error, torch.zeros_like(rgb_error), reduction='sum') / sdf_mask_sum

            psnr = 20.0 * torch.log10(1.0 / (((rgb[sdf_mask] - true_rgb[sdf_mask]) ** 2 * mask[sdf_mask]).sum() / (sdf_mask_sum * 3.0)).sqrt())  # tensor

            loss = rgb_loss + encoder_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', rgb_loss, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
            

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir_mateIllu)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, rgb_loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:  
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0: 
                if self.type == 'dtu' or self.type == 'sk3d':
                    self.validate_image()
                else:
                    self.validate_synthetic_img()
            
            # if self.iter_step % self.val_mesh_freq == 0:  
            #     self.validate_mesh()

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
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)  # 
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:  
            g['lr'] = self.learning_rate * learning_factor  

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'recording'), exist_ok=True)
        for dir_name in dir_lis:  # ['./', './models']
            cur_dir = os.path.join(self.base_exp_dir_mateIllu, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':  
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir_mateIllu, 'recording', 'config.conf'))

    def load_checkpoint_lvis(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir_lvis, 'checkpoints', checkpoint_name), map_location=self.device)
        
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.refColor_network.load_state_dict(checkpoint['refColor_network'])
        self.lvis_network.load_state_dict(checkpoint['lvis_network'])
        self.indiLgt_network.load_state_dict(checkpoint['indiLgt_network'])

        # mateIllu_dict = self.mateIllu_network.state_dict()        
        # net_cd_dict = {k:v for k, v in checkpoint['refColor_network'].items() if k in mateIllu_dict.keys()}
        # mateIllu_dict.update(net_cd_dict)
        # self.mateIllu_network.load_state_dict(mateIllu_dict)

    
    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir_mateIllu, 'checkpoints', checkpoint_name), map_location=self.device)
        
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])   
        self.refColor_network.load_state_dict(checkpoint['refColor_network']) 
        self.lvis_network.load_state_dict(checkpoint['lvis_network'])
        self.indiLgt_network.load_state_dict(checkpoint['indiLgt_network'])

        self.mateIllu_network.load_state_dict(checkpoint['mateIllu_network'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'refColor_network': self.refColor_network.state_dict(),
            'lvis_network': self.lvis_network.state_dict(),
            'indiLgt_network': self.indiLgt_network.state_dict(),

            'mateIllu_network': self.mateIllu_network.state_dict(),

            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir_mateIllu, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
    
    def cal_nerfactor_psnr(self, idx=-1, resolution_level=1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)  # H, W, 3
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)  # H*W/batch, batch, 3
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgba = []
        out_rough = []
        out_albedo = []
        out_normal = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

            render_out = self.renderer.mateIllu_render(rays_o_batch, rays_d_batch, near, far)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('rgb') and feasible('diffuse_albedo') and feasible('roughness'):
                out_rgba.append(render_out['rgb'].detach().cpu().numpy())
                out_albedo.append(render_out['diffuse_albedo'].detach().cpu().numpy())
                out_rough.append(render_out['roughness'].detach().cpu().numpy())
                out_normal.append(render_out['n_out'].detach().cpu().numpy())
            del render_out

        diffuse_albedo_img = np.concatenate(out_albedo, axis=0).reshape([H, W, 3])  # exr

        rgb_img = np.concatenate(out_rgba, axis=0).reshape([H, W, 3])  # exr

        rough_img =  (np.concatenate(out_rough, axis=0).reshape([H, W, 1]) * 255).clip(0, 255)

        n_img = (np.concatenate(out_normal, axis=0).reshape([H, W, 3]) * 128 + 128).clip(0, 255)
        n_img = Image.fromarray(n_img.astype(np.uint8))

        mask = self.dataset.masks[idx, ...].detach().cpu().numpy()  # h,w,3

        tonemap_img = lambda x: np.power(x, 1./2.2)  

        rgb_img = (tonemap_img(rgb_img) * 255).clip(0, 255)
        rgb_img = Image.fromarray(rgb_img.astype(np.uint8))

        diffuse_albedo_img = (tonemap_img(diffuse_albedo_img) * 255).clip(0, 255)
        diffuse_albedo_img = Image.fromarray(diffuse_albedo_img.astype(np.uint8))

        mask = (mask * 255).clip(0, 255)
        mask = Image.fromarray(mask.astype(np.uint8))

        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'psnr'), exist_ok=True)

        rgb_img.save(os.path.join(self.base_exp_dir_mateIllu, 'psnr/preRGB_{}.png'.format(idx)))
        n_img.save(os.path.join(self.base_exp_dir_mateIllu, 'psnr/normal_{}.png'.format(idx)))
        diffuse_albedo_img.save(os.path.join(self.base_exp_dir_mateIllu, 'psnr/preAlbedo_{}.png'.format(idx)))
        mask.save(os.path.join(self.base_exp_dir_mateIllu, 'psnr/mask_{}.png'.format(idx)))

        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'psnr/r_{}.png'.format(idx)), rough_img)

    
    def cal_synthetic_psnr(self, idx=-1, resolution_level=1):
        test_dataset = DatasetSynthetic(self.conf['dataset'], split='test')

        if idx < 0:
            idx = np.random.randint(test_dataset.n_images)

        rays_o, rays_d = test_dataset.gen_rays_at(idx, resolution_level=resolution_level)  # H, W, 3
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)  # H*W/batch, batch, 3
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgba = []
        out_rough = []
        out_albedo = []
        out_normal = []
        out_diffuse_rgb = []
        out_specular_rgb = []
        out_env_rgb = []
        out_indir_rgb = []
        out_lvis_mean = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = test_dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

            render_out = self.renderer.mateIllu_render(rays_o_batch, rays_d_batch, near, far)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('rgb') and feasible('diffuse_albedo') and feasible('roughness'):
                out_rgba.append(render_out['rgb'].detach().cpu().numpy())
                out_albedo.append(render_out['diffuse_albedo'].detach().cpu().numpy())
                out_rough.append(render_out['roughness'].detach().cpu().numpy())
                out_normal.append(render_out['n_out'].detach().cpu().numpy())
                out_env_rgb.append(render_out['env_rgb'].detach().cpu().numpy())
                out_indir_rgb.append(render_out['indir_rgb'].detach().cpu().numpy())
                out_diffuse_rgb.append(render_out['diffuse_rgb'].detach().cpu().numpy())
                out_specular_rgb.append(render_out['specular_rgb'].detach().cpu().numpy())
                out_lvis_mean.append(render_out['lvis_mean'].detach().cpu().numpy())
            del render_out

        tonemap_img = lambda x: np.power(x, 1./2.2)  

        env_rgb_img = np.concatenate(out_env_rgb, axis=0).reshape([H, W, 3])
        env_rgb_img = (tonemap_img(env_rgb_img) * 255).clip(0, 255)
        env_rgb_img = Image.fromarray(env_rgb_img.astype(np.uint8))

        indir_rgb_img = np.concatenate(out_indir_rgb, axis=0).reshape([H, W, 3])
        indir_rgb_img = (tonemap_img(indir_rgb_img) * 255).clip(0, 255)
        indir_rgb_img = Image.fromarray(indir_rgb_img.astype(np.uint8))

        lvis_mean_img = None
        if len(out_lvis_mean) > 0:
            lvis_mean_img = (np.concatenate(out_lvis_mean, axis=0).reshape([H, W, 3]) * 255).clip(0, 255)

        specular_rgb_img = np.concatenate(out_specular_rgb, axis=0).reshape([H, W, 3])
        specular_rgb_img = (tonemap_img(specular_rgb_img) * 255).clip(0, 255)
        specular_rgb_img = Image.fromarray(specular_rgb_img.astype(np.uint8))

        diffuse_albedo_img = np.concatenate(out_albedo, axis=0).reshape([H, W, 3])  # exr
        gt_albedo = test_dataset.albedo[idx].detach().cpu().numpy()  # exr

        rgb_img = np.concatenate(out_rgba, axis=0).reshape([H, W, 3])  # exr
        gt_rgb = test_dataset.images[idx].detach().cpu().numpy()

        rough_img =  (np.concatenate(out_rough, axis=0).reshape([H, W, 1]))
        gt_rough = test_dataset.rough[idx].detach().cpu().numpy()
        gt_rough = gt_rough[..., :1]

        n_img = (np.concatenate(out_normal, axis=0).reshape([H, W, 3]) * 128 + 128).clip(0, 255)
        n_img = Image.fromarray(n_img.astype(np.uint8))

        mask = np.zeros_like(diffuse_albedo_img)
        mask[diffuse_albedo_img > 1e-6] = 1.0

        psnr_albedo = 20.0 * np.log10(1.0 / np.sqrt(((gt_albedo - diffuse_albedo_img) ** 2 * mask).sum() / (mask.sum() * 3.0)))  # tensor
        psnr_rgb = 20.0 * np.log10(1.0 / np.sqrt(((gt_rgb - rgb_img) ** 2 * mask).sum() / (mask.sum() * 3.0)))  # tensor
        psnr_rough = 20.0 * np.log10(1.0 / np.sqrt(((gt_rough - rough_img) ** 2 * mask).sum() / (mask.sum() * 3.0)))  # tensor

        rgb_img = (tonemap_img(rgb_img) * 255).clip(0, 255)
        rgb_img = Image.fromarray(rgb_img.astype(np.uint8))

        diffuse_albedo_img = (tonemap_img(diffuse_albedo_img) * 255).clip(0, 255)
        diffuse_albedo_img = Image.fromarray(diffuse_albedo_img.astype(np.uint8))

        gt_albedo = (tonemap_img(gt_albedo) * 255).clip(0, 255)
        gt_albedo = Image.fromarray(gt_albedo.astype(np.uint8))

        mask = (mask * 255).clip(0, 255)
        mask = Image.fromarray(mask.astype(np.uint8))

        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'psnr'), exist_ok=True)

        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'psnr/lvis_{}_{}.png'.format(self.iter_step, idx)), lvis_mean_img)

        rgb_img.save(os.path.join(self.base_exp_dir_mateIllu, 'psnr/preRGB_{}.png'.format(idx)))
        n_img.save(os.path.join(self.base_exp_dir_mateIllu, 'psnr/normal_{}.png'.format(idx)))
        diffuse_albedo_img.save(os.path.join(self.base_exp_dir_mateIllu, 'psnr/preAlbedo_{}.png'.format(idx)))
        gt_albedo.save(os.path.join(self.base_exp_dir_mateIllu, 'psnr/gtAlbedo_{}.png'.format(idx)))
        mask.save(os.path.join(self.base_exp_dir_mateIllu, 'psnr/mask_{}.png'.format(idx)))

        env_rgb_img.save(os.path.join(self.base_exp_dir_mateIllu, 'psnr/env_rgb_{}.png'.format(idx)))
        indir_rgb_img.save(os.path.join(self.base_exp_dir_mateIllu, 'psnr/indir_rgb_{}.png'.format(idx)))
        specular_rgb_img.save(os.path.join(self.base_exp_dir_mateIllu, 'psnr/specular_rgb_{}.png'.format(idx)))

        rough_img = (rough_img * 255).clip(0, 255)
        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'psnr/r_{}_{}.png'.format(self.iter_step, idx)), rough_img)

        with open(f'{self.base_exp_dir_mateIllu}/psnr/albedo.txt', 'w') as f:
            f.write(f'psnr_albedo:{psnr_albedo}\npsnr_rgb:{psnr_rgb}\npsnr_rough:{psnr_rough}')
    

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

        out_rgb = []
        out_env_rgb = []
        out_indir_rgb = []
        out_diffuse_albedo = []
        out_specular_albedo = []
        out_diffuse_rgb = []
        out_specular_rgb = []
        out_roughness = []
        out_lvis_mean = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

            render_out = self.renderer.mateIllu_render(rays_o_batch, rays_d_batch, near, far)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('rgb') and feasible('diffuse_albedo') and feasible('specular_rgb') and feasible('roughness') and feasible('lvis_mean'):
                out_rgb.append(render_out['rgb'].detach().cpu().numpy())
                out_env_rgb.append(render_out['env_rgb'].detach().cpu().numpy())
                out_indir_rgb.append(render_out['indir_rgb'].detach().cpu().numpy())
                out_diffuse_albedo.append(render_out['diffuse_albedo'].detach().cpu().numpy())
                out_specular_albedo.append(render_out['specular_albedo'].detach().cpu().numpy())
                out_diffuse_rgb.append(render_out['diffuse_rgb'].detach().cpu().numpy())
                out_specular_rgb.append(render_out['specular_rgb'].detach().cpu().numpy())
                out_roughness.append(render_out['roughness'].detach().cpu().numpy())
                out_lvis_mean.append(render_out['lvis_mean'].detach().cpu().numpy())
            del render_out

        tonemap_img = lambda x: np.power(x, 1./2.2)

        rgb_img = None
        if len(out_rgb) > 0:
            rgb_img = np.concatenate(out_rgb, axis=0).reshape([H, W, 3])
            rgb_img = (tonemap_img(rgb_img) * 255).clip(0, 255)

        env_rgb_img = None
        if len(out_env_rgb) > 0:
            env_rgb_img = np.concatenate(out_env_rgb, axis=0).reshape([H, W, 3])
            env_rgb_img = (tonemap_img(env_rgb_img) * 255).clip(0, 255)

        indir_rgb_img = None
        if len(out_indir_rgb) > 0:
            indir_rgb_img = np.concatenate(out_indir_rgb, axis=0).reshape([H, W, 3])
            indir_rgb_img = (tonemap_img(indir_rgb_img) * 255).clip(0, 255)
        
        diffuse_albedo_img = None
        if len(out_diffuse_albedo) > 0:
            diffuse_albedo_img = np.concatenate(out_diffuse_albedo, axis=0).reshape([H, W, 3])
            diffuse_albedo_img = (tonemap_img(diffuse_albedo_img) * 255).clip(0, 255)

        specular_albedo_img = None
        if len(out_specular_albedo) > 0:
            specular_albedo_img = np.concatenate(out_specular_albedo, axis=0).reshape([H, W, 3])
            specular_albedo_img = (tonemap_img(specular_albedo_img) * 255).clip(0, 255)

        diffuse_rgb_img = None
        if len(out_diffuse_rgb) > 0:
            diffuse_rgb_img = np.concatenate(out_diffuse_rgb, axis=0).reshape([H, W, 3])
            diffuse_rgb_img = (tonemap_img(diffuse_rgb_img) * 255).clip(0, 255)

        specular_rgb_img = None
        if len(out_specular_rgb) > 0:
            specular_rgb_img = np.concatenate(out_specular_rgb, axis=0).reshape([H, W, 3])
            specular_rgb_img = (tonemap_img(specular_rgb_img) * 255).clip(0, 255)

        roughness_img = None
        if len(out_roughness) > 0:
            roughness_img = (np.concatenate(out_roughness, axis=0).reshape([H, W, 1]) * 255).clip(0, 255)

        lvis_mean_img = None
        if len(out_lvis_mean) > 0:
            lvis_mean_img = (np.concatenate(out_lvis_mean, axis=0).reshape([H, W, 3]) * 255).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'diffuse'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'specular'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'roughness'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'lvis_mean'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'indi_light'), exist_ok=True)

        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'roughness/r_{}_{}.png'.format(self.iter_step, idx)), roughness_img)
        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'lvis_mean/lvis_{}_{}.png'.format(self.iter_step, idx)), lvis_mean_img)

        indir_rgb_img = Image.fromarray(indir_rgb_img.astype(np.uint8))
        indir_rgb_img.save(os.path.join(self.base_exp_dir_mateIllu, 'indi_light/indiLgt_{}_{}.png'.format(self.iter_step, idx)))

        val_img = np.concatenate([indir_rgb_img, env_rgb_img, rgb_img, self.dataset.image_at(idx, resolution_level=resolution_level)])
        val_img = Image.fromarray(val_img.astype(np.uint8))
        val_img.save(os.path.join(self.base_exp_dir_mateIllu, 'rgb/rgb_{}_{}.png'.format(self.iter_step, idx)))
        
        cd_img = np.concatenate([diffuse_rgb_img, diffuse_albedo_img])
        cd_img = Image.fromarray(cd_img.astype(np.uint8))
        cd_img.save(os.path.join(self.base_exp_dir_mateIllu, 'diffuse/d_{}_{}.png'.format(self.iter_step, idx)))

        diffuse_rgb_img = Image.fromarray(diffuse_rgb_img.astype(np.uint8))
        diffuse_rgb_img.save(os.path.join(self.base_exp_dir_mateIllu, 'diffuse/dc_{}_{}.png'.format(self.iter_step, idx)))
        diffuse_albedo_img = Image.fromarray(diffuse_albedo_img.astype(np.uint8))
        diffuse_albedo_img.save(os.path.join(self.base_exp_dir_mateIllu, 'diffuse/da_{}_{}.png'.format(self.iter_step, idx)))
        
        cs_img = np.concatenate([specular_rgb_img, specular_albedo_img])
        cs_img = Image.fromarray(cs_img.astype(np.uint8))
        cs_img.save(os.path.join(self.base_exp_dir_mateIllu, 'specular/s_{}_{}.png'.format(self.iter_step, idx)))

        specular_rgb_img = Image.fromarray(specular_rgb_img.astype(np.uint8))
        specular_rgb_img.save(os.path.join(self.base_exp_dir_mateIllu, 'specular/sc_{}_{}.png'.format(self.iter_step, idx)))
        specular_albedo_img = Image.fromarray(specular_albedo_img.astype(np.uint8))
        specular_albedo_img.save(os.path.join(self.base_exp_dir_mateIllu, 'specular/sa_{}_{}.png'.format(self.iter_step, idx)))

        env_light = self.mateIllu_network.get_light().detach().cpu().numpy()  # [256,512,3]
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'env_light'), exist_ok=True)
        imageio.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'env_light/iter_step_{}.exr'.format(self.iter_step)), env_light)
    

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

        out_rgb = []
        out_env_rgb = []
        out_indir_rgb = []
        out_diffuse_albedo = []
        out_specular_albedo = []
        out_diffuse_rgb = []
        out_specular_rgb = []
        out_roughness = []
        out_lvis_mean = []

        out_n = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

            render_out = self.renderer.mateIllu_render(rays_o_batch, rays_d_batch, near, far)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('rgb') and feasible('diffuse_albedo') and feasible('specular_rgb') and feasible('roughness') and feasible('lvis_mean'):
                out_rgb.append(render_out['rgb'].detach().cpu().numpy())
                out_env_rgb.append(render_out['env_rgb'].detach().cpu().numpy())
                out_indir_rgb.append(render_out['indir_rgb'].detach().cpu().numpy())
                out_diffuse_albedo.append(render_out['diffuse_albedo'].detach().cpu().numpy())
                out_specular_albedo.append(render_out['specular_albedo'].detach().cpu().numpy())
                out_diffuse_rgb.append(render_out['diffuse_rgb'].detach().cpu().numpy())
                out_specular_rgb.append(render_out['specular_rgb'].detach().cpu().numpy())
                out_roughness.append(render_out['roughness'].detach().cpu().numpy())
                out_lvis_mean.append(render_out['lvis_mean'].detach().cpu().numpy())

                out_n.append(render_out['n_out'].detach().cpu().numpy())
            del render_out

        rgb_img = None
        if len(out_rgb) > 0:
            rgb_img = (np.concatenate(out_rgb, axis=0).reshape([H, W, 3]) * 255).clip(0, 255)
        
        env_rgb_img = None
        if len(out_env_rgb) > 0:
            env_rgb_img = (np.concatenate(out_env_rgb, axis=0).reshape([H, W, 3]) * 255).clip(0, 255)
        
        indir_rgb_img = None
        if len(out_indir_rgb) > 0:
            indir_rgb_img =( np.concatenate(out_indir_rgb, axis=0).reshape([H, W, 3]) * 255).clip(0, 255)

        diffuse_albedo_img = None
        if len(out_diffuse_albedo) > 0:
            diffuse_albedo_img = (np.concatenate(out_diffuse_albedo, axis=0).reshape([H, W, 3]) * 255).clip(0, 255)

        specular_albedo_img1 = None
        if len(out_specular_albedo) > 0:
            specular_albedo_img1 = (np.concatenate(out_specular_albedo, axis=0).reshape([H, W, 3]) * 255).clip(0, 255)

        diffuse_rgb_img = None
        if len(out_diffuse_rgb) > 0:
            diffuse_rgb_img = (np.concatenate(out_diffuse_rgb, axis=0).reshape([H, W, 3]) * 255).clip(0, 255)

        specular_rgb_img = None
        if len(out_specular_rgb) > 0:
            specular_rgb_img = (np.concatenate(out_specular_rgb, axis=0).reshape([H, W, 3]) * 255).clip(0, 255)

        roughness_img = None
        if len(out_roughness) > 0:
            roughness_img = (np.concatenate(out_roughness, axis=0).reshape([H, W, 1]) * 255).clip(0, 255)

        lvis_mean_img = None
        if len(out_lvis_mean) > 0:
            lvis_mean_img = (np.concatenate(out_lvis_mean, axis=0).reshape([H, W, 3]) * 255).clip(0, 255)

        n_img = None
        if len(out_n) > 0:
            n_img = (np.concatenate(out_n, axis=0).reshape([H, W, 3]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'diffuse'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'specular'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'roughness'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'lvis_mean'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'indiLgt'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'normal'), exist_ok=True)

        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'rgb/rgb_{}_{}.png'.format(self.iter_step, idx)), np.concatenate([indir_rgb_img, env_rgb_img, rgb_img, self.dataset.image_at(idx, resolution_level=resolution_level)]))
        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'diffuse/d_{}_{}.png'.format(self.iter_step, idx)), np.concatenate([diffuse_rgb_img, diffuse_albedo_img]))
        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'specular/s_{}_{}.png'.format(self.iter_step, idx)), np.concatenate([specular_rgb_img, specular_albedo_img1]))
        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'roughness/r_{}_{}.png'.format(self.iter_step, idx)), roughness_img)
        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'lvis_mean/lvis_{}_{}.png'.format(self.iter_step, idx)), lvis_mean_img)

        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'specular/sc_{}_{}.png'.format(self.iter_step, idx)), specular_rgb_img)
        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'specular/sa_{}_{}.png'.format(self.iter_step, idx)), specular_albedo_img1)
        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'diffuse/dc_{}_{}.png'.format(self.iter_step, idx)), diffuse_rgb_img)
        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'diffuse/da_{}_{}.png'.format(self.iter_step, idx)), diffuse_albedo_img)
        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'rgb/rgbPre_{}_{}.png'.format(self.iter_step, idx)), rgb_img)
        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'indiLgt/indiLgt_{}_{}.png'.format(self.iter_step, idx)), indir_rgb_img)

        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'normal/n_{}_{}.png'.format(self.iter_step, idx)), n_img)

        env_light = self.mateIllu_network.get_light().detach().cpu().numpy()  # [256,512,3]
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'env_light'), exist_ok=True)
        imageio.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'env_light/iter_step_{}.exr'.format(self.iter_step)), env_light)


    def validate_video(self):
        indir_rgb_list = []
        lvis_mean_list = []
        specular_rgb_img_list = []
        diffuse_rgb_img_list = []
        diffuse_albedo_list = []
        rgb_pre_list = []
        rgb_gt_list = []
        for i in range(self.dataset.n_images):
            rays_o, rays_d = self.dataset.gen_rays_at(img_idx=i, resolution_level=1)  # H, W, 3
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)  # H*W/batch, batch, 3
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

            out_indir_rgb = []
            out_lvis_mean = []
            out_specular_rgb = []
            out_diffuse_rgb = []
            out_diffuse_albedo = []
            out_rgb_pre = []

            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

                render_out = self.renderer.mateIllu_render(rays_o_batch, rays_d_batch, near, far)

                def feasible(key):
                    return (key in render_out) and (render_out[key] is not None)

                if feasible('rgb'):
                    out_indir_rgb.append(render_out['indir_rgb'].detach().cpu().numpy())
                    out_lvis_mean.append(render_out['lvis_mean'].detach().cpu().numpy())
                    out_specular_rgb.append(render_out['specular_rgb'].detach().cpu().numpy())
                    out_diffuse_rgb.append(render_out['diffuse_rgb'].detach().cpu().numpy())
                    out_diffuse_albedo.append(render_out['diffuse_albedo'].detach().cpu().numpy())
                    out_rgb_pre.append(render_out['rgb'].detach().cpu().numpy())
                del render_out

            rgb_img = None
            if len(out_rgb_pre) > 0:
                rgb_img = np.concatenate(out_rgb_pre, axis=0).reshape([H, W, 3])
                rgb_pre_list.append(rgb_img[..., ::-1])  # BGR->RGB

            specular_rgb_img = None
            if len(out_specular_rgb) > 0:
                specular_rgb_img = np.concatenate(out_specular_rgb, axis=0).reshape([H, W, 3])
                specular_rgb_img_list.append(specular_rgb_img[..., ::-1])

            diffuse_rgb_img = None
            if len(out_diffuse_rgb) > 0:
                diffuse_rgb_img = np.concatenate(out_diffuse_rgb, axis=0).reshape([H, W, 3])
                diffuse_rgb_img_list.append(diffuse_rgb_img[..., ::-1])

            diffuse_albedo_img = None
            if len(out_diffuse_albedo) > 0:
                diffuse_albedo_img = np.concatenate(out_diffuse_albedo, axis=0).reshape([H, W, 3])
                diffuse_albedo_list.append(diffuse_albedo_img[..., ::-1])

            indir_rgb_img = None
            if len(out_indir_rgb) > 0:
                indir_rgb_img = np.concatenate(out_indir_rgb, axis=0).reshape([H, W, 3])
                indir_rgb_list.append(indir_rgb_img[..., ::-1])

            lvis_mean_img = None
            if len(out_lvis_mean) > 0:
                lvis_mean_img = np.concatenate(out_lvis_mean, axis=0).reshape([H, W, 3])
                lvis_mean_list.append(lvis_mean_img[..., ::-1])

            rgb_gt_list.append(self.dataset.images[i].detach().cpu().numpy().clip(0., 1.)[..., ::-1])

        rgb_pre_list = rgb_pre_list + [i for i in rgb_pre_list[-2:0:-1]]
        rgb_gt_list = rgb_gt_list + [i for i in rgb_gt_list[-2:0:-1]]
        specular_rgb_img_list = specular_rgb_img_list + [i for i in specular_rgb_img_list[-2:0:-1]]
        diffuse_rgb_img_list = diffuse_rgb_img_list + [i for i in diffuse_rgb_img_list[-2:0:-1]]
        diffuse_albedo_list = diffuse_albedo_list + [i for i in diffuse_albedo_list[-2:0:-1]]
        indir_rgb_list = indir_rgb_list + [i for i in indir_rgb_list[-2:0:-1]]
        lvis_mean_list = lvis_mean_list + [i for i in lvis_mean_list[-2:0:-1]]

        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'video'), exist_ok=True)
        imageio.mimwrite(os.path.join(self.base_exp_dir_mateIllu, 'video/cs.mp4'), specular_rgb_img_list, fps=40, quality=9)
        imageio.mimwrite(os.path.join(self.base_exp_dir_mateIllu, 'video/cd.mp4'), diffuse_rgb_img_list, fps=40, quality=9)
        imageio.mimwrite(os.path.join(self.base_exp_dir_mateIllu, 'video/albedo.mp4'), diffuse_albedo_list, fps=40, quality=9)
        imageio.mimwrite(os.path.join(self.base_exp_dir_mateIllu, 'video/img_pre.mp4'), rgb_pre_list, fps=40, quality=9)
        imageio.mimwrite(os.path.join(self.base_exp_dir_mateIllu, 'video/img_gt.mp4'), rgb_gt_list, fps=40, quality=9)
        imageio.mimwrite(os.path.join(self.base_exp_dir_mateIllu, 'video/indiLgt.mp4'), indir_rgb_list, fps=40, quality=9)
        imageio.mimwrite(os.path.join(self.base_exp_dir_mateIllu, 'video/lvisMean.mp4'), lvis_mean_list, fps=40, quality=9)


    def relgt_synthetic_img(self, idx=-1, resolution_level=1):
        test_dataset = DatasetSynthetic(self.conf['dataset'], split='test')

        rays_o, rays_d = test_dataset.gen_rays_at(idx, resolution_level=resolution_level)  # H, W, 3
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)  # H*W/batch, batch, 3
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        def relgt(name):

            out_rgba = []

            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                near, far = test_dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

                render_out = self.renderer.mateIllu_render(rays_o_batch, rays_d_batch, near, far)

                def feasible(key):
                    return (key in render_out) and (render_out[key] is not None)

                if feasible('rgb') and feasible('diffuse_albedo') and feasible('roughness'):
                    out_rgba.append(render_out['rgb'].detach().cpu().numpy())
                del render_out
            
            tonemap_img = lambda x: np.power(x, 1./2.2)
            rgb_img = np.concatenate(out_rgba, axis=0).reshape([H, W, 3])
            rgb_img = (tonemap_img(rgb_img) * 255).clip(0, 255)
            rgb_img = Image.fromarray(rgb_img.astype(np.uint8))

            os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'video'), exist_ok=True)
            rgb_img.save(os.path.join(self.base_exp_dir_mateIllu, 'video/reLgtRGB_{}.png'.format(name)))

        envmap6_path = './envmaps/envmap6'
        self.mateIllu_network.load_light(envmap6_path)
        relgt('envmap6')
        envmap12_path = './envmaps/envmap12'
        self.mateIllu_network.load_light(envmap12_path)
        relgt('envmap12')


    def relgt_synthetic_video(self):

        test_dataset = DatasetSynthetic(self.conf['dataset'], split='test')

        def relgt(name):
            out_rgb_list = []
            for i in range(test_dataset.n_images):  
                rays_o, rays_d = test_dataset.gen_rays_at(img_idx=i, resolution_level=1)  # H, W, 3
                H, W, _ = rays_o.shape
                rays_o = rays_o.reshape(-1, 3).split(self.batch_size)  # H*W/batch, batch, 3
                rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

                out_rgb = []

                for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                    near, far = test_dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

                    render_out = self.renderer.mateIllu_render(rays_o_batch, rays_d_batch, near, far)

                    def feasible(key):
                        return (key in render_out) and (render_out[key] is not None)

                    if feasible('rgb') and feasible('diffuse_albedo') and feasible('specualr_rgb') and feasible('roughness') and feasible('lvis_mean'):
                        out_rgb.append(render_out['rgb'].detach().cpu().numpy())
                    del render_out

                tonemap_img = lambda x: np.power(x, 1./2.2)

                rgb_img = None
                if len(out_rgb) > 0:
                    rgb_img = np.concatenate(out_rgb, axis=0).reshape([H, W, 3])
                    rgb_img = tonemap_img(rgb_img).clip(0., 1.)
                    out_rgb_list.append(rgb_img)

            os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'video'), exist_ok=True)
            imageio.mimwrite(os.path.join(self.base_exp_dir_mateIllu, f'video/relgt_{name}_img.mp4'), out_rgb_list, fps=20, quality=9)
            del out_rgb_list

        envmap6_path = './envmaps/envmap6'
        self.mateIllu_network.load_light(envmap6_path)
        relgt('envmap6')
        envmap12_path = './envmaps/envmap12'
        self.mateIllu_network.load_light(envmap12_path)
        relgt('envmap12')

    
    def validate_synthetic_video(self):
        out_indir_rgb_list = []
        out_lvis_mean_list = []
        out_img_list = []
        out_diffuse_albedo_list = []
        out_rgb_list = []

        test_dataset = DatasetSynthetic(self.conf['dataset'], split='test')
        for i in range(test_dataset.n_images):  
            rays_o, rays_d = test_dataset.gen_rays_at(img_idx=i, resolution_level=1)  # H, W, 3
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)  # H*W/batch, batch, 3
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

            out_indir_rgb = []
            out_lvis_mean = []
            out_diffuse_albedo = []
            out_rgb = []

            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                near, far = test_dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

                render_out = self.renderer.mateIllu_render(rays_o_batch, rays_d_batch, near, far)

                def feasible(key):
                    return (key in render_out) and (render_out[key] is not None)

                if feasible('rgb'):
                    out_rgb.append(render_out['rgb'].detach().cpu().numpy())
                    out_indir_rgb.append(render_out['indir_rgb'].detach().cpu().numpy())
                    out_lvis_mean.append(render_out['lvis_mean'].detach().cpu().numpy())
                    out_diffuse_albedo.append(render_out['diffuse_albedo'].detach().cpu().numpy())
                del render_out

            tonemap_img = lambda x: np.power(x, 1./2.2)

            rgb_img = None
            if len(out_rgb) > 0:
                rgb_img = np.concatenate(out_rgb, axis=0).reshape([H, W, 3])
                rgb_img = tonemap_img(rgb_img).clip(0., 1.)
                out_rgb_list.append(rgb_img)

            diffuse_albedo_img = None
            if len(out_diffuse_albedo) > 0:
                diffuse_albedo_img = np.concatenate(out_diffuse_albedo, axis=0).reshape([H, W, 3])
                diffuse_albedo_img = tonemap_img(diffuse_albedo_img).clip(0., 1.)
                out_diffuse_albedo_list.append(diffuse_albedo_img)

            indir_rgb_img = None
            if len(out_indir_rgb) > 0:
                indir_rgb_img = np.concatenate(out_indir_rgb, axis=0).reshape([H, W, 3])
                indir_rgb_img = tonemap_img(indir_rgb_img).clip(0., 1.)
                out_indir_rgb_list.append(indir_rgb_img)

            lvis_mean_img = None
            if len(out_lvis_mean) > 0:
                lvis_mean_img = np.concatenate(out_lvis_mean, axis=0).reshape([H, W, 3]).clip(0., 1.)
                out_lvis_mean_list.append(lvis_mean_img)

            out_img_list.append(tonemap_img(test_dataset.rgb_images[i].detach().cpu().numpy()).clip(0., 1.))

        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'video'), exist_ok=True)

        imageio.mimwrite(os.path.join(self.base_exp_dir_mateIllu, 'video/pre_img.mp4'), out_rgb_list, fps=20, quality=9)
        imageio.mimwrite(os.path.join(self.base_exp_dir_mateIllu, 'video/albedo.mp4'), out_diffuse_albedo_list, fps=20, quality=9)
        imageio.mimwrite(os.path.join(self.base_exp_dir_mateIllu, 'video/lvis.mp4'), out_lvis_mean_list, fps=20, quality=9)
        imageio.mimwrite(os.path.join(self.base_exp_dir_mateIllu, 'video/indiLgt.mp4'), out_indir_rgb_list, fps=20, quality=9)
        imageio.mimwrite(os.path.join(self.base_exp_dir_mateIllu, 'video/gt_img.mp4'), out_img_list, fps=20, quality=9)


    def shiny_validate_test(self, idx=-1, resolution_level=-1):
        # test_dataset = DatasetShiny2(self.conf['dataset'], split='test')  
        
        if idx < 0:
            idx = np.random.randint(self.dataset)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level  # 4

        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)  # H, W, 3
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)  # H*W/batch, batch, 3
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb = []
        out_env_rgb = []
        out_indir_rgb = []
        out_diffuse_albedo = []
        out_specular_albedo = []
        out_diffuse_rgb = []
        out_specular_rgb = []
        out_roughness = []
        out_lvis_mean = []

        out_n = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

            render_out = self.renderer.mateIllu_render(rays_o_batch, rays_d_batch, near, far)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('rgb') and feasible('diffuse_albedo') and feasible('specular_rgb') and feasible('roughness') and feasible('lvis_mean'):
                out_rgb.append(render_out['rgb'].detach().cpu().numpy())
                out_env_rgb.append(render_out['env_rgb'].detach().cpu().numpy())
                out_indir_rgb.append(render_out['indir_rgb'].detach().cpu().numpy())
                out_diffuse_albedo.append(render_out['diffuse_albedo'].detach().cpu().numpy())
                out_specular_albedo.append(render_out['specular_albedo'].detach().cpu().numpy())
                out_diffuse_rgb.append(render_out['diffuse_rgb'].detach().cpu().numpy())
                out_specular_rgb.append(render_out['specular_rgb'].detach().cpu().numpy())
                out_roughness.append(render_out['roughness'].detach().cpu().numpy())
                out_lvis_mean.append(render_out['lvis_mean'].detach().cpu().numpy())

                out_n.append(render_out['n_out'].detach().cpu().numpy())
            del render_out

        tonemap_img = lambda x: np.power(x, 1./2.2)

        rgb_img = None
        if len(out_rgb) > 0:
            rgb_img = np.concatenate(out_rgb, axis=0).reshape([H, W, 3])
            rgb_img = (tonemap_img(rgb_img) * 255).clip(0, 255)
            rgb_img = Image.fromarray(rgb_img.astype(np.uint8))
        
        env_rgb_img = None
        if len(out_env_rgb) > 0:
            env_rgb_img = np.concatenate(out_env_rgb, axis=0).reshape([H, W, 3])
            env_rgb_img = (tonemap_img(env_rgb_img) * 255).clip(0, 255)
            env_rgb_img = Image.fromarray(env_rgb_img.astype(np.uint8))

        indir_rgb_img = None
        if len(out_indir_rgb) > 0:
            indir_rgb_img = np.concatenate(out_indir_rgb, axis=0).reshape([H, W, 3])
            indir_rgb_img = (tonemap_img(indir_rgb_img) * 255).clip(0, 255)
            indir_rgb_img = Image.fromarray(indir_rgb_img.astype(np.uint8))
        
        diffuse_albedo_img = None
        if len(out_diffuse_albedo) > 0:
            diffuse_albedo_img = np.concatenate(out_diffuse_albedo, axis=0).reshape([H, W, 3])
            diffuse_albedo_img = (tonemap_img(diffuse_albedo_img) * 255).clip(0, 255)
            diffuse_albedo_img = Image.fromarray(diffuse_albedo_img.astype(np.uint8))

        specular_albedo_img = None
        if len(out_specular_albedo) > 0:
            specular_albedo_img = np.concatenate(out_specular_albedo, axis=0).reshape([H, W, 3])
            specular_albedo_img = (tonemap_img(specular_albedo_img) * 255).clip(0, 255)
            specular_albedo_img = Image.fromarray(specular_albedo_img.astype(np.uint8))

        diffuse_rgb_img = None
        if len(out_diffuse_rgb) > 0:
            diffuse_rgb_img = np.concatenate(out_diffuse_rgb, axis=0).reshape([H, W, 3])
            diffuse_rgb_img = (tonemap_img(diffuse_rgb_img) * 255).clip(0, 255)
            diffuse_rgb_img = Image.fromarray(diffuse_rgb_img.astype(np.uint8))

        specular_rgb_img = None
        if len(out_specular_rgb) > 0:
            specular_rgb_img = np.concatenate(out_specular_rgb, axis=0).reshape([H, W, 3])
            specular_rgb_img = (tonemap_img(specular_rgb_img) * 255).clip(0, 255)
            specular_rgb_img = Image.fromarray(specular_rgb_img.astype(np.uint8))

        roughness_img = None
        if len(out_roughness) > 0:
            roughness_img = (np.concatenate(out_roughness, axis=0).reshape([H, W, 1]) * 255).clip(0, 255)

        lvis_mean_img = None
        if len(out_lvis_mean) > 0:
            lvis_mean_img = (np.concatenate(out_lvis_mean, axis=0).reshape([H, W, 3]) * 255).clip(0, 255)

        n_img = None
        if len(out_n) > 0:
            n_img = (np.concatenate(out_n, axis=0).reshape([H, W, 3]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'diffuse'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'specular'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'roughness'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'lvis_mean'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'normal'), exist_ok=True)

        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'roughness/r_{}_{}.png'.format(self.iter_step, idx)), roughness_img)
        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'lvis_mean/lvis_{}_{}.png'.format(self.iter_step, idx)), lvis_mean_img)
        cv.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'normal/n_{}_{}.png'.format(self.iter_step, idx)), n_img)

        diffuse_albedo_img.save(os.path.join(self.base_exp_dir_mateIllu, 'diffuse/da_{}_{}.png'.format(self.iter_step, idx)))
        specular_albedo_img.save(os.path.join(self.base_exp_dir_mateIllu, 'specular/sa_{}_{}.png'.format(self.iter_step, idx)))
        diffuse_rgb_img.save(os.path.join(self.base_exp_dir_mateIllu, 'diffuse/dc_{}_{}.png'.format(self.iter_step, idx)))
        specular_rgb_img.save(os.path.join(self.base_exp_dir_mateIllu, 'specular/sc_{}_{}.png'.format(self.iter_step, idx)))
        rgb_img.save(os.path.join(self.base_exp_dir_mateIllu, 'rgb/rgbPre_{}_{}.png'.format(self.iter_step, idx)))

        env_light = self.mateIllu_network.get_light().detach().cpu().numpy()  # [256,512,3]
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'env_light'), exist_ok=True)
        imageio.imwrite(os.path.join(self.base_exp_dir_mateIllu, 'env_light/iter_step_{}.exr'.format(self.iter_step)), env_light)


    def validate_mesh(self, world_space=True, resolution=512, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir_mateIllu, 'meshes'), exist_ok=True)

        if world_space:  # false
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        # 导出mesh
        mesh.export(os.path.join(self.base_exp_dir_mateIllu, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')


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

    parser.add_argument('--idx', type=int, default=0)

    args = parser.parse_args()  

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)  

    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.type)  

    if args.mode == 'train':
        runner.train()  
    elif args.mode == 'validate_synthetic_video':
        runner.validate_synthetic_video()
    elif args.mode == 'relgt_synthetic_video':
        runner.relgt_synthetic_video()
    elif args.mode == 'validate_video':
        runner.validate_video()
    elif args.mode == 'validate_image':
        if args.type == 'dtu' or args.type == 'sk3d':
            runner.validate_image(idx=args.idx, resolution_level=1)
        elif args.type == 'synthetic':
            runner.validate_synthetic_img(idx=args.idx, resolution_level=1)
        elif args.type == 'shiny':
            if 'car' in args.case:
                idx = 37  
            elif 'helmet' in args.case:
                idx = 60
            elif 'toaster' in args.case:
                idx = 141
            elif 'teapot' in args.case:
                idx = 199
            elif 'coffee' in args.case:
                idx = 46
            else:
                idx = 0
            runner.shiny_validate_test(idx=idx, resolution_level=1)
    elif args.mode == 'indiSG_psnr':
        if 'hotdog' in args.case:
            idx = 190  
        elif 'jugs' in args.case:
            idx = 0
        else:
            idx = 55
        runner.cal_synthetic_psnr(idx=idx, resolution_level=1)
    elif args.mode == 'relgt_synthetic_img':
        if 'hotdog' in args.case:
            idx = 190  
        elif 'jugs' in args.case:
            idx = 0
        else:
            idx = 55
        runner.relgt_synthetic_img(idx=idx, resolution_level=1)
    

