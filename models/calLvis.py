import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder
from models.inverRender import fibonacci_sphere


def gen_light_z(near, far, n_samples, n):
    z_vals = torch.linspace(0.0, 1.0, n_samples)
    z_vals = near + (far - near) * z_vals  # [n_samples,]
    z_vals = torch.broadcast_to(z_vals, (n, len(z_vals)))  # [n, n_samples]
    return z_vals


def near_far_from_sphere(rays_o, rays_d):
    a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)  # [batch,1]
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)  # [batch,1]
    mid = 0.5 * (-b) / a
    near = mid - 1.0
    far = mid + 1.0
    return near, far


def sample_pdf(bins, weights, n_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # [batch, bins-1]
    cdf = torch.cumsum(pdf, -1)  # [batch, bins-1] 
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


def up_sample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s=64):
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
    #  w = Tα = α*Π(1-α)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

    z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
    return z_samples


def compute_weight(rays_o, rays_d, z_vals, sdf_network, deviation_network):
    batch_size, n_samples = z_vals.shape
    sample_dist = (1 - 0.1) / 32.0
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

    inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # [1,1]         # Single parameter
    inv_s = inv_s.expand(sdf.shape[0], 1)  # [batch_size * n_samples, 1]

    gradients = sdf_network.gradient(pts).squeeze()  # [batch*n_samples, 3]
    
    sdf = sdf.detach()
    inv_s = inv_s.detach()
    gradients = gradients.detach()

    true_cos = (dirs * gradients).sum(-1, keepdim=True)  # [batch*n_samples, 1]
    cos_anneal_ratio = 0.0
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                 F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

    # f: sdf
    estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5  # [batch*n_samples, 1]
    estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5
    # Φ(f*s)
    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)  # [batch*n_samples, 1]
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

    pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
    inside_sphere = (pts_norm < 1.0).float().detach()  # [batch, n_samples]

    # [batch,n_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    weights_inside = weights * inside_sphere
    
    

    return weights, weights_inside


def cal_firHit_rgb(rays_o, rays_d, z_vals, sdf_network, color_network): 
    batch_size, n_samples = z_vals.shape
    sample_dist = (1 - 0.1) / 32.0
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

    hit_rgb = torch.zeros([batch_size, 3])

    pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
    inside_sphere = (pts_norm < 1.0).float().detach()  # [batch, n_samples]
    inside_sphere_mask = (torch.sum(inside_sphere, dim=-1) > 0.0).detach()  # [batch,]

    sdf_input = sdf.reshape(-1, n_samples)  # [n,128]
    tmp = torch.sign(sdf_input) * torch.arange(n_samples, 0, -1).float().to(sdf_input.device).reshape(1, n_samples)  # [n,128]
    min_sdf_val, min_sdf_idx = torch.min(tmp, dim=-1)  # [n,]
    sdf_mask = (min_sdf_val < 0.0) & (min_sdf_idx >= 1) & inside_sphere_mask  # [n,]
    n_sdf_mask = sdf_mask.sum()

    if n_sdf_mask:
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
        f_surf = sdf_network(pts_surf)[:, 1:]

        rays_surf = rays_d[sdf_mask]

        rgb_surf = color_network(pts_surf, n_surf, rays_surf, f_surf)
        hit_rgb[sdf_mask] = rgb_surf

    return hit_rgb, sdf_mask


def compute_light_visibility(surf, normal, sdf_network, deviation_network, color_network, lvis_network, indiLgt_network):
    
    # surf: [batch,3]
    # lgtSG: [batch,128,3] 只有axis部分

    n_lights = 64

    lobes = torch.from_numpy(fibonacci_sphere(n_lights).astype(np.float32)).cuda()  # [128,3]
    lgtSG = lobes[None, ...].repeat(surf.shape[0], 1, 1)

    gt_lvis_hit = torch.zeros((surf.shape[0], n_lights), dtype=torch.float32)  # (n_surf_pts(batch), n_lights)
    gt_rgb_hit = torch.zeros((surf.shape[0], n_lights, 3), dtype=torch.float32)  # (n_surf_pts(batch), n_lights, 3)

    pre_lvis_hit = torch.zeros((surf.shape[0], n_lights), dtype=torch.float32)  # (n_surf_pts(batch), n_lights)
    pre_rgb_hit = torch.zeros((surf.shape[0], n_lights, 3), dtype=torch.float32)  # (n_surf_pts(batch), n_lights, 3)

    inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # [1,1]         # Single parameter
    inv_s = inv_s[0][0]

    lpix_chunk = 8
    for i in range(0, n_lights, lpix_chunk):
        # From surface to lights
        surf2l = lgtSG[:, i:i+lpix_chunk, :]  # (n_surf_pts, lpix_chunk, 3) 
        surf2l = F.normalize(surf2l, dim=-1)
        surf2l_flat = torch.reshape(surf2l, (-1, 3))  # (n_surf_pts * lpix_chunk, 3)

        # 三维坐标点
        surf_rep = torch.tile(surf[:, None, :], (1, surf2l.shape[1], 1))  # (n_surf_pts, lpix_chunk, 3)  
        surf_flat = torch.reshape(surf_rep, (-1, 3))  # (n_surf_pts * lpix_chunk, 3)

        # Save memory by ignoring back-lit points
        lcos = torch.einsum('ijk,ik->ij', surf2l, normal)  # (n_surf_pts, lpix_chunk)
        front_lit = lcos > 0  # (n_surf_pts, lpix_chunk)
        if torch.sum(front_lit.float()) == 0:
            # If there is no point being front lit, this visibility buffer is
            # zero everywhere, so no need to update this slice
            continue

        front_lit_flat = torch.reshape(front_lit, (-1,))  # (n_surf_pts * lpix_chunk)
        surf_flat_frontlit = surf_flat[front_lit_flat]  # (n_frontlit_pairs, 3)
        surf2l_flat_frontlit = surf2l_flat[front_lit_flat]  # (n_frontlit_pairs, 3)

        # coarse sample
        with torch.no_grad():
            z_coarse = gen_light_z(near=0.1, far=0.9, n_samples=512, n=surf2l_flat_frontlit.shape[0])  # [n_frontlit_pairs, 64]
            pts_coarse = surf_flat_frontlit[:, None, :] + surf2l_flat_frontlit[:, None, :] * z_coarse[:, :, None]  # (n_frontlit_pairs, 64, 3)
            pts_coarse_flat = torch.reshape(pts_coarse, (-1, 3))  # (n_frontlit_pairs*64, 3)
            coarse_out = sdf_network(pts_coarse_flat).detach()  # [n_frontlit_pairs*64, 257]
            coarse_sdf = coarse_out[:, :1]  # [n_frontlit_pairs*64, 1]

        # fine sample
        z_fine = up_sample(rays_o=surf_flat_frontlit,  # [n_frontlit_pairs, 64]
                           rays_d=surf2l_flat_frontlit,
                           z_vals=z_coarse,
                           sdf=coarse_sdf,
                           n_importance=32,
                           inv_s=inv_s)
        
        surf_rgb = cal_firHit_rgb(rays_o=surf_flat_frontlit,
                                  rays_d=surf2l_flat_frontlit,
                                  z_vals=z_fine,
                                  sdf_network=sdf_network,
                                  color_network=color_network)  # (n_frontlit_pairs, 3)
        
        weights, weights_inside = compute_weight(rays_o=surf_flat_frontlit,  # [n_frontlit_pairs, 64]
                                                     rays_d=surf2l_flat_frontlit,
                                                     z_vals=z_fine,
                                                     sdf_network=sdf_network,
                                                     deviation_network=deviation_network)
        occu = torch.sum(weights_inside.detach(), -1)  # (n_frontlit_pairs,)

        front_lit_full = torch.zeros(gt_lvis_hit.shape, dtype=torch.bool)
        front_lit_full[:, i:i+lpix_chunk] = front_lit
        gt_lvis_hit[front_lit_full] = 1 - occu

        pre_lvis = lvis_network(surf_flat_frontlit, surf2l_flat_frontlit)  # [n_frontlit_pairs,]
        pre_lvis_hit[front_lit_full] = pre_lvis.reshape(-1)

        gt_rgb_hit[front_lit_full] = surf_rgb

        pre_rgb = indiLgt_network(surf)  # [n, 3] -> [n,24,7]
        pre_rgb = query_indir_illum(pre_rgb, surf2l) # [n, light_chunk, 3] 
        pre_rgb_hit[:, i:i+lpix_chunk, :] = pre_rgb

    gt_lvis_hit = torch.clip(gt_lvis_hit, 0., 1.).detach()  # (n_surf_pts, n_lights) 
    gt_rgb_hit = torch.clip(gt_rgb_hit, 0., 1.).detach()
    
    return {
        'gt_lvis': gt_lvis_hit,
        'pre_lvis': pre_lvis_hit,
        'gt_trace_radiance': gt_rgb_hit,
        'pre_trace_radiance': pre_rgb_hit
    }


def sample_dirs(normals, r_theta, r_phi):
        TINY_NUMBER = 1e-6

        z_axis = torch.zeros_like(normals).cuda()
        z_axis[:, :, 0] = 1

        def norm_axis(x):
            return x / (torch.norm(x, dim=-1, keepdim=True) + TINY_NUMBER)

        normals = norm_axis(normals)  
        U = norm_axis(torch.cross(z_axis, normals))
        V = norm_axis(torch.cross(normals, U))  # [n,1,3]

        r_theta = r_theta.unsqueeze(-1).expand(-1, -1, 3)  # [n,4,3]
        r_phi = r_phi.unsqueeze(-1).expand(-1, -1, 3)  # [n,4,3]
        sample_raydirs = U * torch.cos(r_theta) * torch.sin(r_phi) \
                        + V * torch.sin(r_theta) * torch.sin(r_phi) \
                        + normals * torch.cos(r_phi) # [num_cam, num_samples, 3]
        return sample_raydirs  # [n,4,3] 


def query_indir_illum(lgtSGs, sample_dirs):  # [fir_surf_points,24,7] [fir_surf_points,4,3] 
    nsamp = sample_dirs.shape[1]
    nlobe = lgtSGs.shape[1]
    lgtSGs = lgtSGs.unsqueeze(-3).expand(-1, nsamp, -1, -1)  # [fir_surf_points, 4, 24, 7] 
    sample_dirs = sample_dirs.unsqueeze(-2).expand(-1, -1, nlobe, -1)  # [fir_surf_points, 4, 24, 3] 
    
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))  # [fir_surf_points, 4, 24, 3] 
    lgtSGLambdas = lgtSGs[..., 3:4]  # sharpness
    lgtSGMus = lgtSGs[..., -3:]  # positive values amplitude

    pred_radiance = lgtSGMus * torch.exp(  # [fir_surf_points, 4, 24, 3] 
        lgtSGLambdas * (torch.sum(sample_dirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    pred_radiance = torch.sum(pred_radiance, dim=2)  # [fir_surf_points,4,3] 
    return pred_radiance


def cal_indiLgt(surf, normal, sdf_network, deviation_network, color_network, lvis_network, indiLgt_network):
    nsamp = 4  # 4

    # r_theta_low = (torch.rand(surf.shape[0], 2) / 2.0).cuda() * 2 * np.pi
    # r_theta_high = (torch.rand(surf.shape[0], 2) / 2.0 + 0.5).cuda() * 2 * np.pi

    # rand_z_low = (torch.rand(surf.shape[0], 2) / 2.0).cuda() * 0.95
    # rand_z_high = (torch.rand(surf.shape[0], 2) / 2.0 + 0.5).cuda() * 0.95

    # r_theta = torch.cat([r_theta_low, r_theta_high], dim=-1)
    # rand_z = torch.cat([rand_z_low, rand_z_high], dim=-1)

    r_theta = torch.rand(surf.shape[0], nsamp).cuda() * 2 * np.pi  # [hit_points,4]
    rand_z = torch.rand(surf.shape[0], nsamp).cuda() * 0.95  # [hit_points,4]

    r_phi = torch.asin(rand_z)  # [hit_points,4]
    dirs = sample_dirs(normal[:, None, :], r_theta, r_phi)  # [hit_points,4,3]

    surf_flat_frontlit = surf[:, None, :].repeat(1, nsamp, 1).reshape(-1, 3)
    surf2l_flat_frontlit = dirs.reshape(-1 ,3)

    near, far = near_far_from_sphere(rays_o=surf_flat_frontlit, rays_d=surf2l_flat_frontlit)

    # coarse sample
    with torch.no_grad():
        z_coarse = gen_light_z(near=0.0, far=1.0, n_samples=512, n=surf2l_flat_frontlit.shape[0])  # [n_frontlit_pairs, 64]
        pts_coarse = surf_flat_frontlit[:, None, :] + surf2l_flat_frontlit[:, None, :] * z_coarse[:, :, None]  # (n_frontlit_pairs, 64, 3)
        pts_coarse_flat = torch.reshape(pts_coarse, (-1, 3))  # (n_frontlit_pairs*64, 3)
        coarse_out = sdf_network(pts_coarse_flat).detach()  # [n_frontlit_pairs*64, 257]
        coarse_sdf = coarse_out[:, :1]  # [n_frontlit_pairs*64, 1]

    inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # [1,1]         # Single parameter
    inv_s = inv_s[0][0]

    # fine sample
    z_fine = up_sample(rays_o=surf_flat_frontlit,  # [n_frontlit_pairs, 64]
                        rays_d=surf2l_flat_frontlit,
                        z_vals=z_coarse,
                        sdf=coarse_sdf,
                        n_importance=32,
                        inv_s=inv_s)
    
    trace_radiance, sdf_mask = cal_firHit_rgb(rays_o=surf_flat_frontlit,
                                    rays_d=surf2l_flat_frontlit,
                                    z_vals=z_fine,
                                    sdf_network=sdf_network,
                                    color_network=color_network)  # (n_frontlit_pairs, 3), (n_frontlit_pairs,)
        
    weights, weights_inside = compute_weight(rays_o=surf_flat_frontlit,  # [n_frontlit_pairs, 64]
                                            rays_d=surf2l_flat_frontlit,
                                            z_vals=z_fine,
                                            sdf_network=sdf_network,
                                            deviation_network=deviation_network)
    occu = torch.sum(weights_inside.detach(), -1)  # (n_frontlit_pairs,)
    # occu = sdf_mask.float()  # 1 or 0

    lvis = 1 - occu
    gt_lvis = lvis.reshape(surf.shape[0], nsamp).detach()
    gt_trace_radiance = trace_radiance.reshape(surf.shape[0], nsamp, 3).detach()  # [n,4,3]

    pre_lvis = lvis_network(surf_flat_frontlit, surf2l_flat_frontlit).reshape(surf.shape[0], nsamp)  # [n,4]

    pre_trace_radiance = indiLgt_network(surf)  # [n,24,7]
    pre_trace_radiance = query_indir_illum(pre_trace_radiance, dirs)  # [n,4,3]

    return {
        'gt_lvis': gt_lvis,
        'pre_lvis': pre_lvis,
        'gt_trace_radiance': gt_trace_radiance,
        'pre_trace_radiance': pre_trace_radiance
    }
    
