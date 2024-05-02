import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder
from models import math_utils as utils


class SDFNetwork(nn.Module):  # signed distance function
    def __init__(self,
                 d_in,  # 3
                 d_out,  # 257
                 d_hidden,  # 256
                 n_layers,  # 8
                 skip_in=(4,),
                 multires=0,  # 6
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        # [3->39, 256, 256, 256, 256, 256, 256, 256, 256, 257]
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:  # multires->10
            # multires = 7
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)  # input_ch: 39
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)  # 10
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)  # 

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs, iter_step=0):  # inputs:[batch*n_samples, 3]
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs  # [batch*n_samples, 39]
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))  

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)  # [batch*n_samples, 257]

    def sdf(self, x):  # x:[batch*n_samples, 3]
        # self.forward(x) : [batch*n_samples, 257]
        return self.forward(x)[:, :1]  # [batch*n_samples, 1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)  # [n, 3]
        y = self.sdf(x)  # [n, 1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)  # [n, 1]
        gradients = torch.autograd.grad(  # [n, 3]
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,  # 256
                 mode,  # 'idr'
                 d_in,  # 9 = points+view+normal
                 d_out,  # 3
                 d_hidden,  # 256
                 n_layers,  # 4
                 weight_norm=True,
                 multires_view=0,  # 4
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,  # 4 
                 d_in_view=3,
                 multires=0,  # 10
                 multires_view=0,  # 4
                 output_ch=4,
                 skips=[4],  
                 use_viewdirs=False):  # true
        super(NeRF, self).__init__()
        # 网络的维度
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:  
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:  
            # input_ch_view: 27
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        # Implementation according to the official code release
        # (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        # Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)  
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)

        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)  
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)  
                h = F.relu(h)

            rgb = self.rgb_linear(h)  
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)


class RefColor(nn.Module):
    def __init__(self):
        super(RefColor, self).__init__()

        self.dir_enc_fn = utils.generate_ide_fn(4)

        embedview_fn, input_ch = get_embedder(4)
        self.embedview_fn = embedview_fn

        self.net_cd = nn.Sequential(  
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )

        self.viewdir_mlp = nn.ModuleList(
            [nn.LazyLinear(256)
             for i in range(4)])

        self.net_cs = nn.Sequential(
            nn.LazyLinear(1),
            nn.Sigmoid()
        )


    def forward(self, pts, x, dirs, n):

        normals = utils.l2_normalize(n)
        n_enc = self.embedview_fn(n)
        ref_dirs = utils.reflect(-dirs, normals)
        ref_dirs_enc = self.embedview_fn(ref_dirs)

        diffuse_linear = self.net_cd(torch.cat([pts, n_enc, x], dim=-1))

        inputs_cs = torch.cat([n, pts, ref_dirs_enc, x], dim=-1)
        x2 = inputs_cs
        for i, layer in enumerate(self.viewdir_mlp):
            x2 = layer(x2)
            x2 = torch.nn.functional.relu(x2)
            if i % 4 == 0 and i > 0:
                x2 = torch.cat([x2, inputs_cs], dim=-1)

        specular_linear = self.net_cs(x2)
        specular_linear = specular_linear.repeat(1, 3)

        brdf = specular_linear + diffuse_linear

        rgb = torch.clip(utils.linear_to_srgb(brdf), 0.0, 1.0)

        specular_rgb = torch.clip(utils.linear_to_srgb(specular_linear), 0.0, 1.0)
        diffuse_rgb = torch.clip(utils.linear_to_srgb(diffuse_linear), 0.0, 1.0)


        return {
            'rgb': rgb,
            'specular_rgb': specular_rgb,
            'diffuse_rgb': diffuse_rgb
        }


class Lvis(nn.Module):
    def __init__(self):
        super(Lvis, self).__init__()

        embedview_fn_view, input_ch = get_embedder(4)
        self.embedview_fn_view = embedview_fn_view

        embedview_fn_pts, input_ch = get_embedder(10)
        self.embedview_fn_pts = embedview_fn_pts

        self.lvis = nn.Sequential(  
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, pts, view):
        # view = -ray_d
        view_enc = self.embedview_fn_view(view)
        pts_enc = self.embedview_fn_pts(pts)

        lvis_input = torch.cat([pts_enc, view_enc], dim=-1)
        lvis = self.lvis(lvis_input)

        return lvis


class IndirectLight(nn.Module):
    def __init__(self, 
                 num_lgt_sgs=24):
        super(IndirectLight, self).__init__()

        self.num_lgt_sgs = num_lgt_sgs

        embedview_fn_view, input_ch = get_embedder(4)
        self.embedview_fn_view = embedview_fn_view

        embedview_fn_pts, input_ch = get_embedder(10)
        self.embedview_fn_pts = embedview_fn_pts

        self.indi = nn.Sequential(  
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 144)  # 24*6
        )

    def forward(self, pts):
        pts_enc = self.embedview_fn_pts(pts)
        output = self.indi(pts_enc).reshape(-1, self.num_lgt_sgs, 6)

        lgt_lobes = torch.sigmoid(output[..., :2])  
        theta, phi = lgt_lobes[..., :1] * 2 * np.pi, lgt_lobes[..., 1:2] * 2 * np.pi

        lgt_lobes = torch.cat(  # [n,24,3]
            [torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], dim=-1)
        
        lambda_mu = output[..., 2:]  # [n, 24, 4]
        lambda_mu[..., :1] = torch.sigmoid(lambda_mu[..., :1]) * 30 + 0.1  # sharpness
        lambda_mu[..., 1:] = torch.relu(lambda_mu[..., 1:])  # amplitude

        lgt_sgs = torch.cat([lgt_lobes, lambda_mu], axis=-1)  # [n, 24, 7]

        return lgt_sgs