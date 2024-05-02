import imageio
import os
import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from models.embedder import get_embedder
from models import math_utils as utils
import cv2 as cv


TINY_NUMBER = 1e-6
mode = 'dtu'
# mode = 'synthetic'
if mode == 'synthetic':
    tonemap_img = lambda x: x  
else:
    tonemap_img = lambda x: utils.linear_to_srgb(x)

def compute_envmap(lgtSGs, H, W, upper_hemi=False):
    # same convetion as blender    
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), 
                                     torch.linspace(1.0 * np.pi, -1.0 * np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), 
                                     torch.linspace(1.0 * np.pi, -1.0 * np.pi, W)])
    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), 
                            torch.sin(theta) * torch.sin(phi), 
                            torch.cos(phi)], dim=-1)    # [H, W, 3]
                            
    rgb = render_envmap_sg(lgtSGs, viewdirs)
    envmap = rgb.reshape((H, W, 3))
    return envmap

def render_envmap_sg(lgtSGs, viewdirs):
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,] * len(dots_sh) + [M, 7]).expand(dots_sh + [M, 7])
    
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:]) 
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * \
        (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    return rgb


def norm_axis(x):
    return x / (torch.norm(x, dim=-1, keepdim=True) + TINY_NUMBER)


def compute_energy(lgtSGs):  
    lgtLambda = torch.abs(lgtSGs[:, 3:4]) 
    lgtMu = torch.abs(lgtSGs[:, 4:]) 
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - torch.exp(-2.0 * lgtLambda))
    return energy


def fibonacci_sphere(samples=1):  
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points


def lambda_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2):  
    # assume lambda1 << lambda2
    ratio = lambda1 / (lambda2 + TINY_NUMBER)

    # for insurance
    lobe1 = norm_axis(lobe1)
    lobe2 = norm_axis(lobe2)
    dot = torch.sum(lobe1 * lobe2, dim=-1, keepdim=True)
    tmp = torch.sqrt(ratio * ratio + 1. + 2. * ratio * dot + TINY_NUMBER)
    tmp = torch.min(tmp, ratio + 1.)  

    lambda3 = lambda2 * tmp
    lambda1_over_lambda3 = ratio / (tmp + TINY_NUMBER)
    lambda2_over_lambda3 = 1. / (tmp + TINY_NUMBER)
    diff = lambda2 * (tmp - ratio - 1.)

    final_lobes = lambda1_over_lambda3 * lobe1 + lambda2_over_lambda3 * lobe2  # (λ1*axis1+λ2*axis2)/(λ1+λ2)
    final_lambdas = lambda3  # λ1+λ2
    final_mus = mu1 * mu2 * torch.exp(diff)  # amplitude1 * amplitude2

    return final_lobes, final_lambdas, final_mus


def hemisphere_int(lambda_val, cos_beta):
    lambda_val = torch.clamp(lambda_val, min=TINY_NUMBER)
    
    inv_lambda_val = 1. / (lambda_val + TINY_NUMBER)
    t = torch.sqrt(lambda_val + TINY_NUMBER) * (1.6988 + 10.8438 * inv_lambda_val) / (
                1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val + TINY_NUMBER)

    ### note: for numeric stability
    inv_a = torch.exp(-t)
    mask = (cos_beta >= 0).float()
    inv_b = torch.exp(-t * torch.clamp(cos_beta, min=0.))
    s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b + TINY_NUMBER)
    b = torch.exp(t * torch.clamp(cos_beta, max=0.))
    s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.) + TINY_NUMBER)
    s = mask * s1 + (1. - mask) * s2

    A_b = 2. * np.pi / lambda_val * (torch.exp(-lambda_val) - torch.exp(-2. * lambda_val))
    A_u = 2. * np.pi / lambda_val * (1. - torch.exp(-lambda_val))

    return A_b * (1. - s) + A_u * s


def get_diffuse_visibility(points, normals, VisModel, lgtSGLobes, lgtSGLambdas, nsamp=8):
    ########################################
    # sample dirs according to the light SG
    ########################################

    n_lobe = lgtSGLobes.shape[0]
    n_points = points.shape[0]
    light_dirs = lgtSGLobes.clone().detach().unsqueeze(-2)
    lgtSGLambdas = lgtSGLambdas.clone().detach().unsqueeze(-2)

    # add samples from SG lobes
    z_axis = torch.zeros_like(light_dirs).cuda()
    z_axis[:, :, 2] = 1

    light_dirs = norm_axis(light_dirs) #[num_lobes, 1, 3]
    U = norm_axis(torch.cross(z_axis, light_dirs)) 
    V = norm_axis(torch.cross(light_dirs, U))
    # r_phi depends on the sg sharpness
    sharpness = lgtSGLambdas[:, :, 0]
    sg_range = torch.zeros_like(sharpness).cuda()
    sg_range[:, :] = sharpness.min()
    r_phi_range = torch.arccos((-1.95 * sg_range) / sharpness + 1)
    r_theta = torch.rand(n_lobe, nsamp).cuda() * 2 * np.pi
    r_phi = torch.rand(n_lobe, nsamp).cuda() * r_phi_range

    U = U.expand(-1, nsamp, -1)
    V = V.expand(-1, nsamp, -1)
    r_theta = r_theta.unsqueeze(-1).expand(-1, -1, 3)
    r_phi = r_phi.unsqueeze(-1).expand(-1, -1, 3)

    sample_dir = U * torch.cos(r_theta) * torch.sin(r_phi) \
                + V * torch.sin(r_theta) * torch.sin(r_phi) \
                + light_dirs * torch.cos(r_phi) # [num_lobe, num_sample, 3]
    sample_dir = sample_dir.reshape(-1, 3)

    ########################################
    # visibility
    ########################################
    input_dir = sample_dir.unsqueeze(0).expand(n_points, -1, 3)
    input_p = points.unsqueeze(1).expand(-1, n_lobe * nsamp, 3)
    normals = normals.unsqueeze(1).expand(-1, n_lobe * nsamp, 3)
    # vis = 0 if cos(n, w_i) < 0
    cos_term = torch.sum(normals * input_dir, dim=-1) > TINY_NUMBER

    # batch forward
    batch_size = 100000
    n_mask_dir = input_p[cos_term].shape[0]
    # pred_vis = torch.zeros(n_mask_dir, 2).cuda()
    pred_vis = torch.zeros(n_mask_dir, 1).cuda()
    with torch.no_grad():
        for i, indx in enumerate(torch.split(torch.arange(n_mask_dir).cuda(), batch_size, dim=0)):
            pred_vis[indx] = VisModel(input_p[cos_term][indx], input_dir[cos_term][indx])

    # _, pred_vis = torch.max(pred_vis, dim=-1)
    pred_vis = pred_vis.reshape(-1)
    vis = torch.zeros(n_points, n_lobe * nsamp).cuda()
    vis[cos_term] = pred_vis.float()  # cos_term.shape=vis.shape 
    vis = vis.reshape(n_points, n_lobe, nsamp).permute(1, 2, 0)
    
    sample_dir = sample_dir.reshape(-1, nsamp, 3)
    weight_vis = torch.exp(lgtSGLambdas * (torch.sum(sample_dir * light_dirs, dim=-1, keepdim=True) - 1.))
    
    vis = torch.sum(vis * weight_vis, dim=1) / (torch.sum(weight_vis, dim=1) + TINY_NUMBER)
        
    return vis.detach()


def get_specular_visibility(points, normals, viewdirs, VisModel, lgtSGLobes, lgtSGLambdas, nsamp=24):
    ########################################
    # sample dirs according to the BRDF SG
    ########################################

    # light_dirs = lgtSGLobes.clone().detach().unsqueeze(-2)
    # lgtSGLambdas = lgtSGLambdas.clone().detach().unsqueeze(-2)
    light_dirs = lgtSGLobes.unsqueeze(-2)
    lgtSGLambdas = lgtSGLambdas.unsqueeze(-2)

    n_dot_v = torch.sum(normals * viewdirs, dim=-1, keepdim=True)
    n_dot_v = torch.clamp(n_dot_v, min=0.)
    ref_dir = -viewdirs + 2 * n_dot_v * normals
    ref_dir = ref_dir.unsqueeze(1)
    
    # add samples from BRDF SG lobes
    z_axis = torch.zeros_like(ref_dir).cuda()
    z_axis[:, :, 2] = 1

    U = norm_axis(torch.cross(z_axis, ref_dir))
    V = norm_axis(torch.cross(ref_dir, U))
    # r_phi depends on the sg sharpness
    sharpness = lgtSGLambdas[:, :, 0]
    sharpness = torch.clip(sharpness, min=0.1, max=50)
    sg_range = torch.zeros_like(sharpness).cuda()
    sg_range[:, :] = sharpness.min()
    r_phi_range = torch.arccos((-1.90 * sg_range) / sharpness + 1)
    r_theta = torch.rand(ref_dir.shape[0], nsamp).cuda() * 2 * np.pi
    r_phi = torch.rand(ref_dir.shape[0], nsamp).cuda() * r_phi_range

    U = U.expand(-1, nsamp, -1)
    V = V.expand(-1, nsamp, -1)
    r_theta = r_theta.unsqueeze(-1).expand(-1, -1, 3)
    r_phi = r_phi.unsqueeze(-1).expand(-1, -1, 3)

    sample_dir = U * torch.cos(r_theta) * torch.sin(r_phi) \
                + V * torch.sin(r_theta) * torch.sin(r_phi) \
                + ref_dir * torch.cos(r_phi)

    batch_size = 100000
    input_p = points.unsqueeze(1).expand(-1, nsamp, 3)
    input_dir = sample_dir
    normals = normals.unsqueeze(1).expand(-1, nsamp, 3)
    cos_term = torch.sum(normals * input_dir, dim=-1) > TINY_NUMBER
    n_mask_dir = input_p[cos_term].shape[0]
    pred_vis = torch.zeros(n_mask_dir, 1).cuda()
    with torch.no_grad():
        for i, indx in enumerate(torch.split(torch.arange(n_mask_dir).cuda(), batch_size, dim=0)):
            pred_vis[indx] = VisModel(input_p[cos_term][indx], input_dir[cos_term][indx])

    # _, pred_vis = torch.max(pred_vis, dim=-1)
    pred_vis = pred_vis.reshape(-1)
    vis = torch.zeros(points.shape[0], nsamp).cuda()
    vis[cos_term] = pred_vis.float()

    weight_vis = torch.exp(sharpness * (torch.sum(sample_dir * light_dirs, dim=-1) - 1.))
    inf_idx = torch.isinf(torch.sum(weight_vis, dim=-1))
    inf_sample = weight_vis[inf_idx]

    reset_inf = inf_sample.clone()
    reset_inf[torch.isinf(inf_sample)] = 1.0
    reset_inf[~torch.isinf(inf_sample)] = 0.0
    weight_vis[inf_idx] = reset_inf

    vis = torch.sum(vis * weight_vis, dim=-1) / (torch.sum(weight_vis, dim=-1) + TINY_NUMBER)

    return vis.detach()


def integrate_rgb(normal, final_lobes, final_lambdas, final_mus):
    
    mu_cos = 32.7080
    lambda_cos = 0.0315
    alpha_cos = 31.7003           # brdfSG * illuSG * cosSG
    lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos,
                                                      final_lobes, final_lambdas, final_mus)
    
    dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
    dot1 = torch.clamp(dot1, min=0.)
    dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
    dot2 = torch.clamp(dot2, min=0.)
    rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)  # [dots_shape,128,3]
    rgb = rgb.sum(dim=-2)  # [dots_shape,3]
    rgb = torch.clamp(rgb, min=0., max=1.)

    if torch.isnan(rgb).any():
        print(1)

    return rgb


def render_with_all_sg(points, normal, viewdirs, lgtSGs, specular_reflectance,
                       specular_albedo, roughness, diffuse_albedo, gt_specular_linear, 
                       lvis_network=None, indir_lgtSGs=None):
    
    M = lgtSGs.shape[0]  # 128
    dots_shape = list(normal.shape[:-1]) # [n]

    # direct light
    lgtSGs = lgtSGs.unsqueeze(0).expand(dots_shape + [M, 7])  # [dots_shape, M, 7]
    ret = render_with_sg(points, normal, viewdirs, lgtSGs, specular_reflectance,
                          specular_albedo, roughness, diffuse_albedo, gt_specular_linear, 
                          lvis_network=lvis_network)
    
    # indirct light
    indir_rgb = torch.zeros_like(points)
    if indir_lgtSGs is not None:
        indir_rgb = render_with_sg(points, normal, viewdirs, indir_lgtSGs, specular_reflectance, 
                                    specular_albedo, roughness, diffuse_albedo, gt_specular_linear,
                                    comp_vis=False)['env_rgb']

    env_rgb = ret['env_rgb']
    rgb = torch.clip(tonemap_img(env_rgb + indir_rgb), 0., 1.)
    env_rgb = torch.clip(tonemap_img(env_rgb), 0., 1.)
    indir_rgb = torch.clip(tonemap_img(indir_rgb), 0., 1.)
    ret.update({'rgb': rgb, 'indir_rgb': indir_rgb, 'env_rgb': env_rgb})
    return ret


def render_with_sg(points, normal, viewdirs, lgtSGs, specular_reflectance,
                    specular_albedo, roughness, diffuse_albedo, gt_specular_linear, 
                    comp_vis=True, lvis_network=None):
    '''
    :param points: [batch_size, 3]
    :param normal: [batch_size, 3]; ----> camera; must have unit norm
    :param viewdirs: [batch_size, 3]; ----> camera; must have unit norm
    :param lgtSGs: [batch_size, M, 7]
    :param brdf_amplitude: [batch_size, 3]; 
    :param roughness: [batch_size, 1]; values must be positive
    :param diffuse_albedo: [batch_size, 3]; values must lie in [0,1]
    '''

    M = lgtSGs.shape[1]  # 128
    dots_shape = list(normal.shape[:-1])

    ########################################
    # light
    ########################################

    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER) 
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4]) # sharpness
    origin_lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values

    ########################################
    # specular color
    ########################################

    normal = normal.unsqueeze(-2).expand(dots_shape + [M, 3])  # [dots_shape, M, 3]
    viewdirs = viewdirs.unsqueeze(-2).expand(dots_shape + [M, 3]).detach()  # [dots_shape, M, 3]

    #### NDF
    brdfSGLobes = normal  # use normal as the brdf SG lobes
    inv_roughness_pow4 = 2. / (roughness * roughness * roughness * roughness)  # [dots_shape, 1]
    brdfSGLambdas = inv_roughness_pow4.unsqueeze(1).expand(dots_shape + [M, 1])  # [dots_shape, M, 1] 
    mu_val = (inv_roughness_pow4 / np.pi).expand(dots_shape + [3])  # [dots_shape, 1] ---> [dots_shape, 3]
    brdfSGMus = mu_val.unsqueeze(1).expand(dots_shape + [M, 3])  # [dots_shape, M, 3]

    #### perform spherical warping
    v_dot_lobe = torch.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)  # [dots_shape, M, 1]
    # note: for numeric stability
    v_dot_lobe = torch.clamp(v_dot_lobe, min=0.)
    warpBrdfSGLobes = 2 * v_dot_lobe * brdfSGLobes - viewdirs  # [dots_shape, M, 3] 
    warpBrdfSGLobes = warpBrdfSGLobes / (torch.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + TINY_NUMBER)
    warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + TINY_NUMBER)
    warpBrdfSGMus = brdfSGMus  # [..., M, 3]

    new_half = warpBrdfSGLobes + viewdirs
    new_half = new_half / (torch.norm(new_half, dim=-1, keepdim=True) + TINY_NUMBER)
    v_dot_h = torch.sum(viewdirs * new_half, dim=-1, keepdim=True)
    # note: for numeric stability
    v_dot_h = torch.clamp(v_dot_h, min=0.)

    specular_reflectance = specular_reflectance.unsqueeze(1).expand(dots_shape + [M, 3])  # [dots_shape, M, 3]
    F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

    dot1 = torch.sum(warpBrdfSGLobes * normal, dim=-1, keepdim=True)  # equals <o, n> dot(l,n)
    # note: for numeric stability
    dot1 = torch.clamp(dot1, min=0.)
    dot2 = torch.sum(viewdirs * normal, dim=-1, keepdim=True)  # equals <o, n> dit(v,n)
    # note: for numeric stability
    dot2 = torch.clamp(dot2, min=0.)
    k = (roughness + 1.) * (roughness + 1.) / 8.
    k = k.unsqueeze(1).expand(dots_shape + [M, 1])
    G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
    G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
    G = G1 * G2

    Moi = F * G / (4 * dot1 * dot2 + TINY_NUMBER)
    warpBrdfSGMus = specular_albedo[:, None, :] * warpBrdfSGMus * Moi  # [dots_shape, M, 3]  # 或者限制specular albedo的值大于0.8?
    # warpBrdfSGMus = warpBrdfSGMus * Moi  # [dots_shape, M, 3]

    vis_shadow = torch.zeros(dots_shape[0], 3).cuda()
    if comp_vis:
        # light SG visibility
        light_vis = get_diffuse_visibility(points, normal[:, 0, :], lvis_network,
                                           lgtSGLobes[0], lgtSGLambdas[0], nsamp=32)
        light_vis = light_vis.permute(1, 0).unsqueeze(-1).expand(dots_shape +[M, 3])  # [dots_shape,128,3]

        # # indisg way
        # # BRDF SG visibility
        # brdf_vis = get_specular_visibility(points, normal[:, 0, :], viewdirs[:, 0, :], 
        #                                    lvis_network, warpBrdfSGLobes[:, 0], warpBrdfSGLambdas[:, 0], nsamp=16)
        # brdf_vis = brdf_vis.unsqueeze(-1).unsqueeze(-1).expand(dots_shape + [M, 3])  # [dots_shape,128,3]

        # lgtSGMus = origin_lgtSGMus * brdf_vis  # brdfSG_amplitude * Lvis
        
        # direct multiply
        lgtSGMus = origin_lgtSGMus * light_vis
        
        vis_shadow = torch.mean(light_vis, axis=1).squeeze()
    else:
        lgtSGMus = origin_lgtSGMus

    # multiply with light sg;             brdfSG * illuSG
    final_lobes, final_lambdas, final_mus = lambda_trick(lgtSGLobes, lgtSGLambdas, lgtSGMus,  
                                                         warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus)
    
    # Σ(brdfSG * illuSG * cosSG)
    specular_linear = integrate_rgb(normal, final_lobes, final_lambdas, final_mus)  # linear
    # specular_linear = specular_linear * specular_albedo

    ########################################
    # diffuse color
    ########################################

    if comp_vis:
        lgtSGMus = origin_lgtSGMus * light_vis
    else:
        lgtSGMus = origin_lgtSGMus 

    diffuse = (diffuse_albedo / np.pi).unsqueeze(-2).expand(dots_shape + [M, 3])  # [dots_shape,128,3]
    # multiply with light sg
    final_lobes = lgtSGLobes
    final_lambdas = lgtSGLambdas
    final_mus = lgtSGMus * diffuse  # amplitude = SG_amplitude * Lvis * diffuse

    diffuse_linear = integrate_rgb(normal, final_lobes, final_lambdas, final_mus)

    ########################################
    # color
    ########################################

    rgb = torch.clamp(specular_linear + diffuse_linear, 0.0, 1.0) 

    ret = {
        'specular_loss': 0,
        'diffuse_loss': 0,
        'env_rgb': rgb,
        'diffuse_rgb': torch.clip(tonemap_img(diffuse_linear), 0., 1.),
        'specular_rgb': torch.clip(tonemap_img(specular_linear), 0., 1.),
        'lvis_mean': vis_shadow
        # 'lvis_mean': torch.mean(light_vis, dim=-1, keepdim=True)  # [dot_shape, 1]
    }

    return ret

class EnvmapMaterialNetwork(nn.Module):
    def __init__(self,
                 num_lgt_sgs=128, 
                 specular_albedo=0.02):  # 0.02 0.75
        super().__init__()

        self.numLgtSGs = num_lgt_sgs

        embed_view_fn, input_ch = get_embedder(4)
        self.embed_view_fn = embed_view_fn
        
        embed_pts_fn, input_ch = get_embedder(10)
        self.embed_pts_fn = embed_pts_fn

        embed_dot_fn, input_ch = get_embedder(4, input_dims=1)
        self.embed_dot_fn = embed_dot_fn

        ############################################### encoder decoder #############################################################

        self.brdf_embed_fn, brdf_input_dim = get_embedder(10)
        brdf_encoder_dims=[512, 512, 512, 512]
        brdf_decoder_dims=[128, 128]
        self.latent_dim = 32
        self.actv_fn = nn.LeakyReLU(0.2)

        brdf_encoder_layer = []  # 63->512*4->32
        dim = brdf_input_dim
        for i in range(len(brdf_encoder_dims)):
            brdf_encoder_layer.append(nn.Linear(dim, brdf_encoder_dims[i]))
            brdf_encoder_layer.append(self.actv_fn)
            dim = brdf_encoder_dims[i]
        brdf_encoder_layer.append(nn.Linear(dim, self.latent_dim))
        self.brdf_encoder_layer = nn.Sequential(*brdf_encoder_layer)
        
        brdf_decoder_layer = []  # 32->128*2->4(roughness+diffuse_albedo)
        dim = self.latent_dim
        for i in range(len(brdf_decoder_dims)):
            brdf_decoder_layer.append(nn.Linear(dim, brdf_decoder_dims[i]))
            brdf_decoder_layer.append(self.actv_fn)
            dim = brdf_decoder_dims[i]
        brdf_decoder_layer.append(nn.Linear(dim, 4))
        # brdf_decoder_layer.append(nn.Linear(dim, 3))  # 4->3
        self.brdf_decoder_layer = nn.Sequential(*brdf_decoder_layer)     

        self.net_cs = nn.Sequential( 
            nn.LazyLinear(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 没有sigmoid，rough会变白
        )

        ########## fresnel ##########
        self.specular_reflectance = torch.zeros([1,1])
        self.specular_reflectance[:] = specular_albedo

        ########## light SG #########
        self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 7), requires_grad=True) # [M, 7]; lobe + lambda + mu
        self.lgtSGs.data[:, -2:] = self.lgtSGs.data[:, -3:-2].expand((-1, 2))  
        # make sure lambda is not too close to zero
        self.lgtSGs.data[:, 3:4] = 10. + torch.abs(self.lgtSGs.data[:, 3:4] * 20.) 
        # init envmap energy
        energy = compute_energy(self.lgtSGs.data)  # [128,7] -> [128,3]
        self.lgtSGs.data[:, 4:] = torch.abs(self.lgtSGs.data[:, 4:]) / torch.sum(energy, dim=0, keepdim=True) * 2. * np.pi * 0.8  
        energy = compute_energy(self.lgtSGs.data)  # [128,7] -> [128,3]

        # deterministicly initialize lobes; 
        lobes = fibonacci_sphere(self.numLgtSGs//2).astype(np.float32)  
        self.lgtSGs.data[:self.numLgtSGs//2, :3] = torch.from_numpy(lobes)
        self.lgtSGs.data[self.numLgtSGs//2:, :3] = torch.from_numpy(lobes)

        # used for relighting
        self.envmap = None

    def forward(self, points, ray_dirs, n, f, gt_specular_linear, indiLgt, lvis_network):
        ###############################
        # pre
        ###############################
        n = n / (torch.norm(n, dim=-1, keepdim=True) + TINY_NUMBER)
        ray_dirs = ray_dirs / (torch.norm(ray_dirs, dim=-1, keepdim=True) + TINY_NUMBER)
        
        view_dirs = -ray_dirs
        ref_dirs = utils.reflect(view_dirs, n)
        
        n_enc = self.embed_view_fn(n)
        view_enc = self.embed_view_fn(view_dirs)
        ref_dirs_enc = self.embed_view_fn(ref_dirs)
        pts_enc = self.embed_pts_fn(points)

        ########################################## encoder decoder ######################################################

        points_enc = self.brdf_embed_fn(points)
        brdf_lc = torch.sigmoid(self.brdf_encoder_layer(points_enc))  # [n,32]
        brdf = torch.sigmoid(self.brdf_decoder_layer(brdf_lc))  # [n,4]
        roughness = brdf[..., 3:] * 0.9 + 0.09  # [n,1]
        diffuse_albedo = brdf[..., :3]  # [n,3]
        # diffuse_albedo = brdf
        # roughness = self.net_roughness(points_enc) * 0.9 + 0.09

        # kl loss
        values = points_enc
        for i in range(len(self.brdf_encoder_layer)):
            values = self.brdf_encoder_layer[i](values)
        loss = 0.0
        loss += self.kl_divergence(0.05, values)
        loss = 0.01 * loss

        # specualr albedo
        specular_albedo_input = torch.cat([pts_enc, ref_dirs_enc], dim=-1)  
        specular_albedo = self.net_cs(specular_albedo_input).repeat(1, 3)  
        
        ret = render_with_all_sg(points, n, view_dirs, self.lgtSGs, self.specular_reflectance,
                                  specular_albedo, roughness, diffuse_albedo, gt_specular_linear, 
                                  lvis_network=lvis_network, indir_lgtSGs=indiLgt)
        

        # smooth loss
        # rand_points = points + torch.randn(points.shape).cuda() * 0.01
        # rand_points_enc = self.brdf_embed_fn(rand_points)
        # rand_brdf_lc = torch.sigmoid(self.brdf_encoder_layer(rand_points_enc))
        # rand_brdf = torch.sigmoid(self.brdf_decoder_layer(rand_brdf_lc))
        # random_xi_roughness = rand_brdf[..., 3:] * 0.9 + 0.09
        # random_xi_diffuse_albedo = rand_brdf[..., :3]
        
        # ret_random = render_with_all_sg2(points, n, view_dirs, self.lgtSGs, self.specular_reflectance,
        #                           specular_albedo, random_xi_roughness, random_xi_diffuse_albedo, gt_specular_linear, 
        #                           lvis_network=lvis_network, indir_lgtSGs=indiLgt)
        
        # smooth_loss = Func.l1_loss(ret_random['specular_rgb'], ret['specular_rgb'], reduction='mean')
        # smooth_loss = 0.01 * smooth_loss  # 0.1->0.01
        smooth_loss = 0.
        
        

        ret.update({
            'roughness': roughness, 
            'diffuse_albedo': torch.clip(tonemap_img(diffuse_albedo), 0., 1.),
            'specular_albedo': torch.clip(tonemap_img(specular_albedo), 0., 1.),
            'encoder_loss': loss,
            'smooth_loss': smooth_loss
            })
        
        return ret
    
    def get_light(self):
        # lgtSGs = self.lgtSGs.clone().detach()
        # limit lobes to upper hemisphere
        # if self.upper_hemi:
        #     lgtSGs = self.restrict_lobes_upper(lgtSGs)
        lgtSGs = self.lgtSGs
        envmap = compute_envmap(lgtSGs=lgtSGs, H=256, W=512)  # [256,512,3]
        return envmap
    
    def kl_divergence(self, rho, rho_hat):
        rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
        rho = torch.tensor([rho] * len(rho_hat)).cuda()
        return torch.mean(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

    def load_light(self, path):
        sg_path = os.path.join(path, 'sg_128.npy')
        device = self.lgtSGs.data.device
        load_sgs = torch.from_numpy(np.load(sg_path)).to(device)  # [128,7]
        self.lgtSGs.data = load_sgs

        energy = compute_energy(self.lgtSGs.data)  # [128,3]
        print('loaded envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

        envmap_path = path + '.exr'
        envmap = np.float32(imageio.imread(envmap_path)[:, :, :3])  # (512, 1024, 3)
        self.envmap = torch.from_numpy(envmap).to(device)
    
    














