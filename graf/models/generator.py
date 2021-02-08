import numpy as np
import torch
from torch import nn
from ..utils import sample_on_sphere, look_at, to_sphere
from ..transforms import FullRaySampler
from ...submodules.nerf_pytorch.run_nerf_mod import render, run_network
from functools import partial

class Generator(nn.Module):
    def __init__(self, H, W, focal, radius, ray_sampler, render_kwargs_train,
            render_kwargs_test, parameters, named_parameters,
                 range_u=(0,1), range_v=(0.01,0.49), chunk=None,
                 device='cuda', orthographic=False):
        super(Generator, self).__init__()
        self.device = device
        self.H = int(H)
        self.W = int(W)
        self.focal = focal
        self.radius = radius
        self.range_u = range_u
        self.range_v = range_v
        self.chunk = chunk
        coords = torch.from_numpy(np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), -1))
        self.coords = coords.view(-1, 2)

        self.ray_sampler = ray_sampler
        self.val_ray_sampler = FullRaySampler(orthographic=orthographic)
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.train_net = render_kwargs_train.pop('network_fn')
        self.train_net_fine = render_kwargs_train.pop('network_fine')
        self.test_net = render_kwargs_test.pop('network_fn')
        self.test_net_fine = render_kwargs_test.pop('network_fine')
        self.initial_raw_noise_std = self.render_kwargs_train['raw_noise_std']
        self.render = partial(render, H=self.H, W=self.W, focal=self.focal, chunk=self.chunk)

    def forward(self, z, y=None, rays=None):
        bs = z.shape[0]
        if rays is None:
            rays = torch.cat([self.sample_rays() for _ in range(bs)], dim=1)

        render_kwargs = self.render_kwargs_train if self.training\
                else self.render_kwargs_test
        render_kwargs = dict(render_kwargs)        # copy
    
        # in the case of a variable radius
        # we need to adjust near and far plane for the rays
        # so they stay within the bounds defined wrt. maximal radius
        # otherwise each camera samples within its own near/far plane (relative to this camera's radius)
        # instead of the absolute value (relative to maximum camera radius)
        if isinstance(self.radius, tuple):
            assert self.radius[1] - self.radius[0] <= render_kwargs['near'], 'Your smallest radius lies behind your near plane!'
    
            rays_radius = rays[0].norm(dim=-1)
            shift = (self.radius[1] - rays_radius).view(-1, 1).float()      # reshape s.t. shape matches required shape in run_nerf
            render_kwargs['near'] = render_kwargs['near'] - shift
            render_kwargs['far'] = render_kwargs['far'] - shift
            assert (render_kwargs['near'] >= 0).all() and (render_kwargs['far'] >= 0).all(), \
                (rays_radius.min(), rays_radius.max(), shift.min(), shift.max())
            

        render_kwargs['features'] = z
        net, net_fine = (self.train_net, self.train_net_fine) if\
                self.training else (self.test_net, self.test_net_fine)
        rgb, disp, acc, extras = render(self.H, self.W, self.focal,
                chunk=self.chunk, rays=rays, network_fn=net,
                network_fine=net_fine, **render_kwargs)

        rays_to_output = lambda x: x.view(len(x), -1) * 2 - 1      # (BxN_samples)xC
    
        if not self.training:               # return all outputs
            return rays_to_output(rgb), \
                   rays_to_output(disp), \
                   rays_to_output(acc), extras

        rgb = rays_to_output(rgb)
        return rgb

    def decrease_nerf_noise(self, it):
        end_it = 5000
        if it < end_it:
            noise_std = self.initial_raw_noise_std - self.initial_raw_noise_std/end_it * it
            self.render_kwargs_train['raw_noise_std'] = noise_std

    def sample_pose(self):
        # sample location on unit sphere
        loc = sample_on_sphere(self.range_u, self.range_v)
        
        # sample radius if necessary
        radius = self.radius
        if isinstance(radius, tuple):
            radius = np.random.uniform(*radius)

        loc = loc * radius
        R = look_at(loc)[0]

        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        RT = torch.Tensor(RT.astype(np.float32)).to(self.device)
        return RT

    def sample_rays(self):
        pose = self.sample_pose()
        sampler = self.ray_sampler if self.training else self.val_ray_sampler
        batch_rays, _, _ = sampler(self.H, self.W, self.focal, pose)
        return batch_rays
