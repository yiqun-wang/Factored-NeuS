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
        
        self.base_exp_dir_lvis = self.conf['general.base_exp_dir_lvis']  
        self.base_exp_dir_geometry = self.conf['general.base_exp_dir_geo']
        os.makedirs(self.base_exp_dir_lvis, exist_ok=True)

        self.type = type
        if self.type == 'dtu':
            self.dataset = Dataset(self.conf['dataset'])
        elif self.type == 'sk3d':
            self.dataset = DatasetSk3d(self.conf['dataset'])
        elif self.type == 'synthetic':
            self.dataset = DatasetSynthetic(self.conf['dataset'])
        elif self.type == 'shiny':
            self.dataset = DatasetShiny(self.conf['dataset'])

        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.lvis.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')  
        self.val_freq = self.conf.get_int('train.val_freq')

        self.batch_size = self.conf.get_int('train.lvis.batch_size')  # 512

        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.lvis.warm_up_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []  

        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)  # color
        self.refColor_network = RefColor().to(self.device)
        
        self.lvis_network = Lvis().to(self.device)
        self.indiLgt_network = IndirectLight().to(self.device)
        self.mateIllu_network = EnvmapMaterialNetwork().to(self.device)

        params_to_train += list(self.lvis_network.parameters())
        params_to_train += list(self.indiLgt_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        model_list_raw = os.listdir(os.path.join(self.base_exp_dir_geometry, 'checkpoints'))
        model_list = []
        for model_name in model_list_raw:
            if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.conf.get_int('train.end_iter'):
                model_list.append(model_name)
        model_list.sort()
        latest_model_name = model_list[-1]
        logging.info('Find geometry checkpoint: {}'.format(latest_model_name))
        self.load_checkpoint_geometry(latest_model_name)

        self.renderer = NeuSRenderer(**self.conf['model.lvis_renderer'],
                                     sdf_network=self.sdf_network,
                                     deviation_network=self.deviation_network,
                                     color_network=self.color_network,
                                     lvis_network=self.lvis_network,
                                     indiLgt_network=self.indiLgt_network,
                                     mateIllu_network=self.mateIllu_network
                                    )
        
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir_lvis, 'checkpoints'))
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
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir_lvis, 'logs'))  
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

            lvis_out = self.renderer.lvis_render(rays_o, rays_d, near, far)

            gt_lvis = lvis_out['gt_lvis']
            pre_lvis = lvis_out['pre_lvis']  # [512,4]

            gt_trace_radiance = lvis_out['gt_trace_radiance']
            pre_trace_radiance = lvis_out['pre_trace_radiance']  # [512,4,3]

            sdf_mask = lvis_out['sdf_mask']
            if sdf_mask.sum() < 1e-6:
                continue  

            # loss
            lvis_error = (gt_lvis - pre_lvis)
            lvis_loss = F.l1_loss(lvis_error, torch.zeros_like(lvis_error), reduction='sum') / (sdf_mask[..., None].expand(gt_lvis.shape).sum() + 1e-6)

            trace_radiance_error = (gt_trace_radiance - pre_trace_radiance) * sdf_mask[..., None, None]
            trace_radiance_loss = F.l1_loss(trace_radiance_error, torch.zeros_like(trace_radiance_error), reduction='sum') / (sdf_mask[..., None, None].expand(gt_trace_radiance.shape).sum() + 1e-6)

            loss = lvis_loss + trace_radiance_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', lvis_loss, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir_lvis)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, lvis_loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0: 
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:  
                if self.type == 'dtu' or self.type == 'sk3d':
                    self.validate_image()
                else:
                    self.validate_synthetic_img()

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
        os.makedirs(os.path.join(self.base_exp_dir_lvis, 'recording'), exist_ok=True)
        for dir_name in dir_lis:  # ['./', './models']
            cur_dir = os.path.join(self.base_exp_dir_lvis, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py': 
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir_lvis, 'recording', 'config.conf'))

    def load_checkpoint_geometry(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir_geometry, 'checkpoints', checkpoint_name), map_location=self.device)
        
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.refColor_network.load_state_dict(checkpoint['refColor_network'])
    
    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir_lvis, 'checkpoints', checkpoint_name), map_location=self.device)
        
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])   
        self.color_network.load_state_dict(checkpoint['color_network_fine']) 
        self.refColor_network.load_state_dict(checkpoint['refColor_network'])

        self.lvis_network.load_state_dict(checkpoint['lvis_network'])
        self.indiLgt_network.load_state_dict(checkpoint['indiLgt_network'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'refColor_network': self.refColor_network.state_dict(),

            'lvis_network': self.lvis_network.state_dict(),
            'indiLgt_network': self.indiLgt_network.state_dict(),

            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir_lvis, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir_lvis, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
    
    
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

        out_gt_lvis = []
        out_pre_lvis = []
        out_gt_trace_radiance = []
        out_pre_trace_radiance = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

            render_out = self.renderer.lvis_render(rays_o_batch, rays_d_batch, near, far)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('gt_lvis') and feasible('pre_lvis') and feasible('gt_trace_radiance') and feasible('pre_trace_radiance'):
                out_gt_lvis.append(render_out['gt_lvis'].detach().cpu().numpy())
                out_pre_lvis.append(render_out['pre_lvis'].detach().cpu().numpy())
                out_gt_trace_radiance.append(render_out['gt_trace_radiance'].detach().cpu().numpy())
                out_pre_trace_radiance.append(render_out['pre_trace_radiance'].detach().cpu().numpy())
            del render_out

        tonemap_img = lambda x: np.power(x, 1./2.2)

        nsamp = 4

        gt_trace_radiance_img = None
        if len(out_gt_trace_radiance) > 0:
            gt_trace_radiance_img = np.concatenate(out_gt_trace_radiance, axis=0).reshape([H, W, nsamp, 3])
            gt_trace_radiance_img = (tonemap_img(gt_trace_radiance_img) * 255).clip(0, 255)
        
        pre_trace_radiance_img = None
        if len(out_pre_trace_radiance) > 0:
            pre_trace_radiance_img = np.concatenate(out_pre_trace_radiance, axis=0).reshape([H, W, nsamp, 3])
            pre_trace_radiance_img = (tonemap_img(pre_trace_radiance_img) * 255).clip(0, 255)

        gt_lvis_img = None
        if len(out_gt_lvis) > 0:
            gt_lvis_img = np.concatenate(out_gt_lvis, axis=0).reshape([H, W, nsamp])
            gt_lvis_img = np.mean(gt_lvis_img, axis=-1, keepdims=True)
            gt_lvis_img = (gt_lvis_img * 255).clip(0, 255)
        
        pre_lvis_img = None
        if len(out_pre_lvis) > 0:
            pre_lvis_img = np.concatenate(out_pre_lvis, axis=0).reshape([H, W, nsamp])
            pre_lvis_img = np.mean(pre_lvis_img, axis=-1, keepdims=True)
            pre_lvis_img = (pre_lvis_img * 255).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir_lvis, 'trace_radiance/{}').format(self.iter_step), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_lvis, 'lvis'), exist_ok=True)

        cv.imwrite(os.path.join(self.base_exp_dir_lvis, 'lvis/lvis_{}_{}.png'.format(self.iter_step, idx)), np.concatenate([pre_lvis_img, gt_lvis_img]))

        trace_radiance_img_mean = np.concatenate([np.mean(pre_trace_radiance_img, axis=-2), np.mean(gt_trace_radiance_img, axis=-2)])
        trace_radiance_img_mean = Image.fromarray(trace_radiance_img_mean.astype(np.uint8))
        trace_radiance_img_mean.save(os.path.join(self.base_exp_dir_lvis, 'trace_radiance/{}/trace_radiance_mean_{}_{}.png'.format(self.iter_step, self.iter_step, idx)))
        
        # for i in range(nsamp):
        #     trace_radiance_img = np.concatenate([pre_trace_radiance_img[..., i, :], gt_trace_radiance_img[..., i, :]])
        #     trace_radiance_img = Image.fromarray(trace_radiance_img.astype(np.uint8))
        #     trace_radiance_img.save(os.path.join(self.base_exp_dir_lvis, 'trace_radiance/{}/trace_radiance_{}_{}_{}.png'.format(self.iter_step, self.iter_step, i, idx)))

    
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

        out_gt_lvis = []
        out_pre_lvis = []
        out_gt_trace_radiance = []
        out_pre_trace_radiance = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            
            render_out = self.renderer.lvis_render(rays_o_batch, rays_d_batch, near, far)
            
            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('gt_lvis') and feasible('pre_lvis') and feasible('gt_trace_radiance') and feasible('pre_trace_radiance'):
                out_gt_lvis.append(render_out['gt_lvis'].detach().cpu().numpy())
                out_pre_lvis.append(render_out['pre_lvis'].detach().cpu().numpy())
                out_gt_trace_radiance.append(render_out['gt_trace_radiance'].detach().cpu().numpy())
                out_pre_trace_radiance.append(render_out['pre_trace_radiance'].detach().cpu().numpy())
            del render_out

        nsamp = 4

        gt_trace_radiance_img = None
        if len(out_gt_trace_radiance) > 0:
            gt_trace_radiance_img = np.concatenate(out_gt_trace_radiance, axis=0).reshape([H, W, nsamp, 3])
            gt_trace_radiance_img = np.mean(gt_trace_radiance_img, axis=-2)
            gt_trace_radiance_img = (gt_trace_radiance_img * 255).clip(0, 255)

        pre_trace_radiance_img = None
        if len(out_pre_trace_radiance) > 0:
            pre_trace_radiance_img = np.concatenate(out_pre_trace_radiance, axis=0).reshape([H, W, nsamp, 3])
            pre_trace_radiance_img = np.mean(pre_trace_radiance_img, axis=-2)
            pre_trace_radiance_img = (pre_trace_radiance_img * 255).clip(0, 255)

        gt_lvis_img = None
        if len(out_gt_lvis) > 0:
            gt_lvis_img = np.concatenate(out_gt_lvis, axis=0).reshape([H, W, nsamp])
            gt_lvis_img = np.mean(gt_lvis_img, axis=-1, keepdims=True)
            gt_lvis_img = (gt_lvis_img * 255).clip(0, 255)
        
        pre_lvis_img = None
        if len(out_pre_lvis) > 0:
            pre_lvis_img = np.concatenate(out_pre_lvis, axis=0).reshape([H, W, nsamp])
            pre_lvis_img = np.mean(pre_lvis_img, axis=-1, keepdims=True)
            pre_lvis_img = (pre_lvis_img * 255).clip(0, 255)
        
        os.makedirs(os.path.join(self.base_exp_dir_lvis, 'trace_radiance'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir_lvis, 'lvis'), exist_ok=True)

        cv.imwrite(os.path.join(self.base_exp_dir_lvis, 'trace_radiance/trace_radiance{}_{}.png'.format(self.iter_step, idx)), np.concatenate([pre_trace_radiance_img, gt_trace_radiance_img]))
        cv.imwrite(os.path.join(self.base_exp_dir_lvis, 'lvis/lvis_{}_{}.png'.format(self.iter_step, idx)), np.concatenate([pre_lvis_img, gt_lvis_img]))
                


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

    args = parser.parse_args()  

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)  

    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.type)  

    if args.mode == 'train':
        runner.train()  
    elif args.mode == 'validate_image':
        if args.type == 'dtu' or args.type == 'sk3d':
            runner.validate_image(resolution_level=1)
        elif args.type == 'synthetic':
            runner.validate_synthetic_img(resolution_level=1)

