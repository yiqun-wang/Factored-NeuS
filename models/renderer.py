import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic
from collections import OrderedDict
from models import math_utils as utils

from models.calLvis import cal_indiLgt, compute_light_visibility


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)  # 1, 64
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)  # [64, 64, 64]
    with torch.no_grad():  
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)  # [64, 64, 64]
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [64^3, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()  # [64, 64, 64]
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)  # [64, 64, 64]
    vertices, triangles = mcubes.marching_cubes(u, threshold)  # [vertices or triangles, 3]
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    '''
    bins: z_vals(通过远近面均匀划分的采样t0...tn) [batch, n_samples(64)]
    weights: [batch, n_samples-1]
    n_samples: n_importance (16)
    det: true
    '''
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # [batch, bins-1]
    cdf = torch.cumsum(pdf, -1)  # [batch, bins-1] [:,-1]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # [batch, bins]
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])  # [batch, n_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()  
    inds = torch.searchsorted(cdf, u, right=True)  # [batch,n_samples]
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)  # [batch,n_samples]
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)  # [batch,n_samples]
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]  # [batch,n_samples,bins]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)  # [batch,n_samples,2]
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)  # [batch,n_samples,2]
    denom = (cdf_g[..., 1] - cdf_g[..., 0])  # [batch,n_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom  # [batch,n_samples]
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])  # [batch,n_samples]
    return samples


class NeuSRenderer:
    def __init__(self,
                 n_samples,  # 64
                 n_importance,  # 64
                 n_outside,  # 32
                 up_sample_steps,  # 4
                 perturb,  # 1.0
                 nerf=None,
                 sdf_network=None,
                 deviation_network=None,
                 color_network=None,
                 refColor_network=None,
                 lvis_network=None,
                 indiLgt_network=None,
                 mateIllu_network=None
                 ): 
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network

        self.refColor_network = refColor_network
        self.lvis_network = lvis_network
        self.indiLgt_network = indiLgt_network
        self.mateIllu_network = mateIllu_network

        self.n_samples = n_samples  # 64
        self.n_importance = n_importance  # 64
        self.n_outside = n_outside  # 32
        self.up_sample_steps = up_sample_steps  # 4
        self.perturb = perturb  # 1.0

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        batch_size, n_samples = z_vals.shape  # n_samples:64(uniform)+64(importance)+32(outside)=160

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [batch, n_samples-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)  # [batch, n_samples]
        mid_z_vals = z_vals + dists * 0.5  # [batch, n_samples]

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # [batch_size, n_samples, 3]
        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)  # [batch_size, n_samples, 4]

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)  # [batch_size, n_samples, 3]

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))  # [batch_size*n_samples, 4]
        dirs = dirs.reshape(-1, 3)  # [batch_size*n_samples, 3]

        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color)
        # α = 1-exp(σδ)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)  # [batch_size, n_samples]
        alpha = alpha.reshape(batch_size, n_samples)
        # w = α*T = α*Π(1-α)
        # [batch_size, n_samples]
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)  # [batch_size, n_samples, 3]
        # C = Σwc
        color = (weights[:, :, None] * sampled_color).sum(dim=1)  # [batch, 3]
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }


    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        n_importance: 16
        inv_s: 64
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # [batch, n_samples, 3]
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)  # [batch, n_samples]
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)  # [batch, n_samples-1]
        sdf = sdf.reshape(batch_size, n_samples)  # [batch, n_samples]
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]  # [batch, n_samples-1]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]  # [batch, n_samples-1]
        mid_sdf = (prev_sdf + next_sdf) * 0.5  # [batch, n_samples-1]
        # ±|cos(θ)| = df/dt 
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)  # [batch, n_samples-1]
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)  # [batch, n_samples-1]
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)  # [batch, n_samples-1, 2]
        # min(cos,pre_cos)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)  # [batch, n_samples-1]
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        # f: sdf
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        # Φ(f)=sigmoid(s*f)
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        # α = (next_Φ-prev_Φ)/prev_Φ
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        # equation 11 w = Tα = α*Π(1-α)
        # [batch, 63]
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)  # [batch,n_samples+n_importance]
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)  # [batch,n_importance]
            sdf = torch.cat([sdf, new_sdf], dim=-1)  # [batch,n_samples+n_importance]
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    #
    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    refColor_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape  # 512， 128

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)  # [batch, n_samples]
        mid_z_vals = z_vals + dists * 0.5

        mid_dists = mid_z_vals[..., 1:] - mid_z_vals[..., :-1]
        mid_dists = torch.cat([mid_dists, torch.Tensor([sample_dist]).expand(mid_dists[..., :1].shape)], -1)

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)  # [batch, n_samples, 3]

        pts = pts.reshape(-1, 3)  # [batch*n_samples, 3]
        dirs = dirs.reshape(-1, 3)  # [batch*n_samples, 3]

        sdf_nn_output = sdf_network(pts)  # [batch*n_samples, 257]
        sdf = sdf_nn_output[:, :1]  # [batch*n_samples, 1]
        feature_vector = sdf_nn_output[:, 1:]  # [batch*n_samples, 256]

        gradients = sdf_network.gradient(pts).squeeze()  # [batch*n_samples, 3]
        # sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # [1,1]         # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)  # [batch_size * n_samples, 1]

        true_cos = (dirs * gradients).sum(-1, keepdim=True)  # [batch*n_samples, 1]

        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        # f: sdf
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5  # [batch*n_samples, 1]
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5
        # Φ(f*s)
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)  # [batch*n_samples, 1]
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        # sigma
        cdf = torch.sigmoid(sdf * inv_s)
        sigma = inv_s * (1 - cdf) * (-iter_cos) * mid_dists.reshape(-1, 1)

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()  # [batch, n_samples]
        relax_inside_sphere = (pts_norm < 1.2).float().detach()  # [batch, n_samples]

        inside_sphere_mask = (torch.sum(inside_sphere, dim=-1) > 0.0).detach()  # [batch,]
        
        ######################################################################################################################

        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        specular_color = torch.ones([batch_size, 3])
        diffuse_color = torch.ones([batch_size, 3])
        surface_color = torch.ones([batch_size, 3])        

        pts_input = pts.reshape(-1, n_samples, 3)
        normals_input = gradients.reshape(-1, n_samples, 3)
        dirs_input = dirs.reshape(-1, n_samples, 3)
        feature_vector_input = feature_vector.reshape(-1, n_samples, 256)
        sdf_input = sdf.reshape(-1, n_samples)  # [n,128]

        tmp = torch.sign(sdf_input) * torch.arange(n_samples, 0, -1).float().to(sdf_input.device).reshape(1, n_samples)  # [n,128]
        min_sdf_val, min_sdf_idx = torch.min(tmp, dim=-1)  # [n,]
        sdf_mask = (min_sdf_val < 0.0) & (min_sdf_idx >= 1) & inside_sphere_mask  # [n,]
        n_sdf_mask = sdf_mask.sum()

        # if n_sdf_mask > 0 and iter_step > 50000:
        if n_sdf_mask > 0:
            min_sdf_idx_feature = min_sdf_idx[sdf_mask].reshape(-1, 1, 1).repeat(1, 1, 256)
            min_sdf_idx_weight = min_sdf_idx[sdf_mask].reshape(-1, 1)
            min_sdf_idx = min_sdf_idx[sdf_mask].reshape(-1, 1, 1).repeat(1, 1, 3)

            pts_input_low = torch.gather(pts_input[sdf_mask], dim=1, index=min_sdf_idx - 1)  # [mask, 1, 3]
            pts_input_high = torch.gather(pts_input[sdf_mask], dim=1, index=min_sdf_idx)  # [mask, 1, 3]
            pts_input_total = torch.cat([pts_input_low, pts_input_high], dim=1).reshape(-1, 3)  # [mask*2, 3]

            normals_input_low = torch.gather(normals_input[sdf_mask], dim=1, index=min_sdf_idx - 1)  # [mask, 1, 3]
            normals_input_high = torch.gather(normals_input[sdf_mask], dim=1, index=min_sdf_idx)  # [mask, 1, 3]
            normals_input_total = torch.cat([normals_input_low, normals_input_high], dim=1).reshape(-1, 3)  # [mask*2, 3]

            dirs_input_low = torch.gather(dirs_input[sdf_mask], dim=1, index=min_sdf_idx - 1)  # [mask, 1, 3]
            dirs_input_high = torch.gather(dirs_input[sdf_mask], dim=1, index=min_sdf_idx)  # [mask, 1, 3]
            dirs_input_total = torch.cat([dirs_input_low, dirs_input_high], dim=1).reshape(-1, 3)  # [mask*2, 3]

            feature_vector_input_low = torch.gather(feature_vector_input[sdf_mask], dim=1,
                                                    index=min_sdf_idx_feature - 1)  # [mask, 1, 256]
            feature_vector_input_high = torch.gather(feature_vector_input[sdf_mask], dim=1,
                                                     index=min_sdf_idx_feature)  # [mask, 1, 256]
            feature_input_total = torch.cat([feature_vector_input_low, feature_vector_input_high], dim=1).reshape(-1, 256)  # [mask*2, 256]

            ref_dict = refColor_network(pts_input_total,
                                        feature_input_total,
                                        dirs_input_total,
                                        normals_input_total)

            rgb = ref_dict['rgb'].reshape(-1, 2, 3)
            specular = ref_dict['specular_rgb'].reshape(-1, 2, 3)
            diffuse = ref_dict['diffuse_rgb'].reshape(-1, 2, 3)

            alpha_inside = alpha * inside_sphere
            weights_inside = alpha_inside * torch.cumprod(  # [batch, n_samples]
                torch.cat([torch.ones([batch_size, 1]), 1. - alpha_inside + 1e-7], -1), -1)[:, :-1]
            weights_inside_low = torch.gather(weights_inside[sdf_mask], dim=1, index=min_sdf_idx_weight - 1) + 1e-5  # [mask, 1]
            weights_inside_high = torch.gather(weights_inside[sdf_mask], dim=1, index=min_sdf_idx_weight) + 1e-5  # [mask, 1]

            specular = (specular[:, 0, :] * weights_inside_low + specular[:, 1, :] * weights_inside_high) / (   # [mask, 3]
                        weights_inside_low + weights_inside_high)
            diffuse = (diffuse[:, 0, :] * weights_inside_low + diffuse[:, 1, :] * weights_inside_high) / (   # [mask, 3]
                        weights_inside_low + weights_inside_high)
            rgb = (rgb[:, 0, :] * weights_inside_low + rgb[:, 1, :] * weights_inside_high) / (   # [mask, 3]
                        weights_inside_low + weights_inside_high)  
            
            specular_color[sdf_mask] = specular
            diffuse_color[sdf_mask] = diffuse
            surface_color[sdf_mask] = rgb

        #####################################################################################################################

        # Render with background
        # background_alpha=128(uniform+importance)+32(outside)
        # background_alpha = None
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)  # [batch, uniform+importance+outside]

            sampled_color = sampled_color * inside_sphere[:, :, None] + \
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)  # [batch, 160]

        # w = αΠ(1-α)
        # [batch, 160(uniform+importance+outside)]
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        # C = Σwc
        color = (sampled_color * weights[:, :, None]).sum(dim=1)  # [batch, 3]
        # background_rgb = None
        if background_rgb is not None:  # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss 
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2  # batch, n_samples
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)  # tensor

        return {
            'color': color,  # [batch, 3] uniform+importance+outside 
            'surface_color': surface_color,
            'sdf_mask': sdf_mask,
            'sdf': sdf,  # [batch*n_samples, 1] uniform+importance 
            'dists': dists,  # [batch, n_samples] uniform+importance 
            'gradients': gradients.reshape(batch_size, n_samples, 3),  # uniform+importance 
            's_val': 1.0 / inv_s,  # [batch*n_samples, 1] s
            'mid_z_vals': mid_z_vals,  # [batch, n_samples] : [t0...tn]
            'weights': weights,  # [batch, uniform+importance+outside]
            'cdf': c.reshape(batch_size, n_samples),  # [batch*n_samples, 1] 
            'gradient_error': gradient_error,  # tensor
            'inside_sphere': inside_sphere,  # [batch, n_samples]
            'specular_color': specular_color,
            'diffuse_color': diffuse_color
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples  # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]  # [batch(512),n_samples(64)]

        z_vals_outside = None
        if self.n_outside > 0: 
            # [batch(512),n_outside(32)]
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:  
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples  # [batch(512),n_samples(64)]

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])  # [31,]
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)  # [n_outside(32),]
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)  # [n_outside(32),]
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])  # [batch(512),n_outside(32)]
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand  # [batch(512),n_outside(32)] 

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                # o+td
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # [batch(512),n_samples(64),3]
                # [batch*n_samples(32768), 3(xyz)] -> [batch*n_samples(32768), 1(sdf)] -> [batch(512),n_samples(64)]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                z_vals_fine = []
                for i in range(self.up_sample_steps):
                    # [batch,16]
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,  # 64//4
                                                64 * 2 ** i)
                    z_vals_fine.append(new_z_vals)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance  # 128

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)  # [batch,64+64+32]
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']  # [batch, n_samples+n_importance+n_outside, 3]
            background_alpha = ret_outside['alpha']  # [batch, n_samples+n_importance+n_outside (160)]

        # Render core 
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    self.refColor_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']

        surface_color = ret_fine['surface_color']
        specular_color = ret_fine['specular_color']
        diffuse_color = ret_fine['diffuse_color']
        sdf_mask = ret_fine['sdf_mask']

        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)  # [batch, 1]

        return {
            'color_fine': color_fine,
            'surface_color': surface_color,
            'sdf_mask': sdf_mask,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'specular_color': specular_color,
            'diffuse_color': diffuse_color
        }
    

    def lvis_mateIllu_render_util(self, rays_o, rays_d, near, far):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples  # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]  # [batch(512),n_samples(64)]

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                # o+td
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # [batch(512),n_samples(64),3]
                # [batch*n_samples(32768), 3(xyz)] -> [batch*n_samples(32768), 1(sdf)] -> [batch(512),n_samples(64)]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)
                z_vals_fine = []
                for i in range(self.up_sample_steps):
                    # [batch,16]
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,  # 64//4
                                                64 * 2 ** i)
                    z_vals_fine.append(new_z_vals)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance  # 128

        # batch_size, n_samples = z_vals.shape  # 512， 128
        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)  # [batch, n_samples]
        mid_z_vals = z_vals + dists * 0.5

        mid_dists = mid_z_vals[..., 1:] - mid_z_vals[..., :-1]
        mid_dists = torch.cat([mid_dists, torch.Tensor([sample_dist]).expand(mid_dists[..., :1].shape)], -1)

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)  # [batch, n_samples, 3]

        pts = pts.reshape(-1, 3)  # [batch*n_samples, 3]
        dirs = dirs.reshape(-1, 3)  # [batch*n_samples, 3]

        sdf_nn_output = self.sdf_network(pts)  # [batch*n_samples, 257]
        sdf = sdf_nn_output[:, :1]  # [batch*n_samples, 1]

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()  # [batch, n_samples]

        inside_sphere_mask = (torch.sum(inside_sphere, dim=-1) > 0.0).detach()  # [batch,]

        return {
            'n_samples': n_samples,
            'mid_z_vals': mid_z_vals,
            'sdf': sdf,
            'inside_sphere_mask': inside_sphere_mask
        }
    

    def lvis_render(self, rays_o, rays_d, near, far):
        batch_size = len(rays_o)
        sdf_network = self.sdf_network
        deviation_network = self.deviation_network
        lvis_network = self.lvis_network

        util_res = self.lvis_mateIllu_render_util(rays_o, rays_d, near, far)    
        n_samples = util_res['n_samples']    
        mid_z_vals = util_res['mid_z_vals']
        sdf = util_res['sdf']
        inside_sphere_mask = util_res['inside_sphere_mask']

        #######################################################################################
        M = 4

        gt_lvis_out = torch.ones([batch_size, M])
        pre_lvis_out = torch.ones([batch_size, M])

        gt_trace_radiance_out = torch.ones([batch_size, M, 3])
        pre_trace_radiance_out = torch.ones([batch_size, M, 3])

        sdf_input = sdf.reshape(-1, n_samples)  # [batch,128]

        tmp = torch.sign(sdf_input) * torch.arange(n_samples, 0, -1).float().to(sdf_input.device).reshape(1, n_samples)  # [batch,128]
        min_sdf_val, min_sdf_idx = torch.min(tmp, dim=-1)  # [batch,]
        sdf_mask = (min_sdf_val < 0.0) & (min_sdf_idx >= 1) & inside_sphere_mask  # [batch,]
        n_sdf_mask = sdf_mask.sum()

        if n_sdf_mask > 0:
            min_z_vals_idx = min_sdf_idx[sdf_mask].reshape(-1, 1)  # [mask, 1]
            # [batch, n_samples] -> [mask, n_samples] -> [mask, 1]
            z_vals_low = torch.gather(mid_z_vals[sdf_mask], dim=1, index=min_z_vals_idx - 1)  # [mask, 1]
            z_vals_high = torch.gather(mid_z_vals[sdf_mask], dim=1, index=min_z_vals_idx)  # [mask, 1]
            sdf_low = torch.gather(sdf_input[sdf_mask], dim=1, index=min_z_vals_idx - 1)  # [mask, 1]
            sdf_high = torch.gather(sdf_input[sdf_mask], dim=1, index=min_z_vals_idx)  # [mask, 1]
            z_vals_surf = (sdf_low * z_vals_high - sdf_high * z_vals_low) / (sdf_low - sdf_high + 1e-10)  # [mask, 1]
            pts_surf = rays_o[sdf_mask][:, None, :] + rays_d[sdf_mask][:, None, :] * z_vals_surf[..., :, None]  # [mask,1,3]
            pts_surf = pts_surf.reshape(-1, 3)  # [mask, 3]
            n_surf = sdf_network.gradient(pts_surf).squeeze().reshape(-1, 3)  # [mask, 3] 
            
            cal_res = cal_indiLgt(pts_surf, n_surf, sdf_network, deviation_network, self.color_network, lvis_network, self.indiLgt_network)  # [mask,4]
            # cal_res = compute_light_visibility(pts_surf, n_surf, sdf_network, deviation_network, self.color_network, lvis_network, self.indiLgt_network)  # [mask,4]

            gt_lvis = cal_res['gt_lvis']  # [mask, 4]
            pre_lvis = cal_res['pre_lvis']  # [mask, 4]
            gt_trace_radiance = cal_res['gt_trace_radiance']  # [mask,4,3]
            pre_trace_radiance = cal_res['pre_trace_radiance']  # [mask,4,3]

            gt_lvis_out[sdf_mask] = gt_lvis
            pre_lvis_out[sdf_mask] = pre_lvis

            gt_trace_radiance_out[sdf_mask] = gt_trace_radiance
            pre_trace_radiance_out[sdf_mask] = pre_trace_radiance

        return {
            'gt_lvis': gt_lvis_out,  # [n, 4]
            'pre_lvis': pre_lvis_out,  # [n, 4]
            'gt_trace_radiance': gt_trace_radiance_out,  # [n,4,3]
            'pre_trace_radiance': pre_trace_radiance_out,  # [n,4,3]
            'sdf_mask': sdf_mask
        }
    

    def mateIllu_render(self, rays_o, rays_d, near, far):
        batch_size = len(rays_o)
        util_res = self.lvis_mateIllu_render_util(rays_o, rays_d, near, far)    
        n_samples = util_res['n_samples']    
        mid_z_vals = util_res['mid_z_vals']
        sdf = util_res['sdf']
        inside_sphere_mask = util_res['inside_sphere_mask']

        rgb = torch.ones([batch_size, 3])
        env_rgb = torch.ones([batch_size, 3])
        indir_rgb = torch.ones([batch_size, 3])
        diffuse_albedo = torch.ones([batch_size, 3])
        specular_albedo = torch.ones([batch_size, 3])
        diffuse_rgb = torch.ones([batch_size, 3])
        specular_rgb = torch.ones([batch_size, 3])
        roughness = torch.ones([batch_size, 1])
        lvis_mean = torch.ones([batch_size, 3])

        n_out = torch.ones([batch_size, 3])

        gt_specular_linear = torch.ones([batch_size, 3])
        gt_diffuse_srgb = torch.ones([batch_size, 3])

        diffuse_loss = 0
        specular_loss = 0
        encoder_loss = 0
        smooth_loss = 0

        sdf_input = sdf.reshape(-1, n_samples)  # [batch,128]
        tmp = torch.sign(sdf_input) * torch.arange(n_samples, 0, -1).float().to(sdf_input.device).reshape(1, n_samples)  # [batch,128]
        min_sdf_val, min_sdf_idx = torch.min(tmp, dim=-1)  # [batch,]
        sdf_mask = (min_sdf_val < 0.0) & (min_sdf_idx >= 1) & inside_sphere_mask  # [batch,]
        n_sdf_mask = sdf_mask.sum()

        if n_sdf_mask > 0:
            min_z_vals_idx = min_sdf_idx[sdf_mask].reshape(-1, 1)  # [mask, 1]
            # [batch, n_samples] -> [mask, n_samples] -> [mask, 1]
            z_vals_low = torch.gather(mid_z_vals[sdf_mask], dim=1, index=min_z_vals_idx - 1)  # [mask, 1]
            z_vals_high = torch.gather(mid_z_vals[sdf_mask], dim=1, index=min_z_vals_idx)  # [mask, 1]
            sdf_low = torch.gather(sdf_input[sdf_mask], dim=1, index=min_z_vals_idx - 1)  # [mask, 1]
            sdf_high = torch.gather(sdf_input[sdf_mask], dim=1, index=min_z_vals_idx)  # [mask, 1]
            z_vals_surf = (sdf_low * z_vals_high - sdf_high * z_vals_low) / (sdf_low - sdf_high + 1e-10)  # [mask, 1]
            pts_surf = rays_o[sdf_mask][:, None, :] + rays_d[sdf_mask][:, None, :] * z_vals_surf[..., :, None]  # [mask,1,3]
            pts_surf = pts_surf.reshape(-1, 3)  # [mask, 3]
            n_surf = self.sdf_network.gradient(pts_surf).squeeze().reshape(-1, 3)  # [mask, 3] 
            f_surf = self.sdf_network(pts_surf)[:, 1:]

            rays_surf = rays_d[sdf_mask]

            refColor_res = self.refColor_network(pts_surf, f_surf, rays_surf, n_surf)            
            diffuse_srgb = refColor_res['diffuse_rgb']
            specular_srgb = refColor_res['specular_rgb']
            diffuse_linear = utils.srgb_to_linear(diffuse_srgb)
            specular_linear = utils.srgb_to_linear(specular_srgb)

            indiLgt = self.indiLgt_network(pts_surf)
            mateIllu_out = self.mateIllu_network(pts_surf, rays_surf, n_surf, f_surf, specular_linear, indiLgt, self.lvis_network)

            env_rgb[sdf_mask] = mateIllu_out['env_rgb']
            indir_rgb[sdf_mask] = mateIllu_out['indir_rgb']
            rgb[sdf_mask] = mateIllu_out['rgb']
            diffuse_albedo[sdf_mask] = mateIllu_out['diffuse_albedo']
            specular_albedo[sdf_mask] = mateIllu_out['specular_albedo']
            diffuse_rgb[sdf_mask] = mateIllu_out['diffuse_rgb']
            specular_rgb[sdf_mask] = mateIllu_out['specular_rgb']
            roughness[sdf_mask] = mateIllu_out['roughness']
            lvis_mean[sdf_mask] = mateIllu_out['lvis_mean']

            gt_specular_linear[sdf_mask] = specular_linear
            gt_diffuse_srgb[sdf_mask] = diffuse_srgb

            diffuse_loss = mateIllu_out['diffuse_loss']
            specular_loss = mateIllu_out['specular_loss']
            encoder_loss = mateIllu_out['encoder_loss']
            smooth_loss = mateIllu_out['smooth_loss']

            n_out[sdf_mask] = n_surf
    
        return {
            'rgb': rgb, 
            'env_rgb': env_rgb,
            'indir_rgb': indir_rgb,
            'diffuse_albedo': diffuse_albedo,
            'specular_albedo': specular_albedo,
            'diffuse_rgb': diffuse_rgb,
            'specular_rgb': specular_rgb,
            'roughness': roughness,
            'lvis_mean': lvis_mean,
            'sdf_mask': sdf_mask,
            'diffuse_loss': diffuse_loss,
            'specular_loss': specular_loss,
            'encoder_loss': encoder_loss,
            'smooth_loss': smooth_loss,
            'gt_specular_linear': gt_specular_linear,
            'gt_diffuse_srgb': gt_diffuse_srgb,
            'n_out': n_out
        }


    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
