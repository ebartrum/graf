import numpy as np
import torch
from torchvision.transforms import *
from .datasets import *
from .transforms import FlexGridRaySampler
from .utils import polar_to_cartesian, look_at, to_phi, to_theta
from .models.generator import Generator
from .models.discriminator import Discriminator
from ..submodules.nerf_pytorch.run_nerf_mod import create_nerf
from argparse import Namespace

def save_config(outpath, config):
    from yaml import safe_dump
    with open(outpath, 'w') as f:
        safe_dump(config, f)


def update_config(config, unknown):
    # update config given args
    for idx,arg in enumerate(unknown):
        if arg.startswith("--"):
            if (':') in arg:
                k1,k2 = arg.replace("--","").split(':')
                argtype = type(config[k1][k2])
                if argtype == bool:
                    v = unknown[idx+1].lower() == 'true'
                else:
                    if config[k1][k2] is not None:
                        v = type(config[k1][k2])(unknown[idx+1])
                    else:
                        v = unknown[idx+1]
                print(f'Changing {k1}:{k2} ---- {config[k1][k2]} to {v}')
                config[k1][k2] = v
            else:
                k = arg.replace('--','')
                v = unknown[idx+1]
                argtype = type(config[k])
                print(f'Changing {k} ---- {config[k]} to {v}')
                config[k] = v

    return config

def get_dataset(config, transform):
    H = W = imsize = config['data']['imsize']
    dset_type = config['data']['type']
    fov = config['data']['fov']

    kwargs = {
        'data_dirs': config.root,
        'mask_dir': config['data']['maskdir'] if 'maskdir' in config['data'].keys() else None,
        'transforms': transform
    }

    if dset_type == 'carla':
        dset = Carla(**kwargs)

    elif dset_type == 'celebA':
        assert imsize <= 128, 'cropped GT data has lower resolution than imsize, consider using celebA_hq instead'
        transform.transforms.insert(0, RandomHorizontalFlip())
        transform.transforms.insert(0, CenterCrop(108))

        dset = CelebA(**kwargs)

    elif dset_type == 'celebA_hq':
        transform.transforms.insert(0, RandomHorizontalFlip())
        transform.transforms.insert(0, CenterCrop(650))

        dset = CelebAHQ(**kwargs)

    elif dset_type == 'cats':
      transform.transforms.insert(0, RandomHorizontalFlip())
      dset = Cats(**kwargs)
  
    elif dset_type == 'cub':
        dset = CUB(**kwargs)

    elif dset_type == 'image_folder':
        if config['data']['crop'] > 0:
            transform.transforms.insert(0, CenterCrop(config['data']['crop']))
        dset = ImageDataset(data_dirs=glob.glob(kwargs['data_dirs']+'/*'),
                mask_dir=kwargs['mask_dir'],
                transforms=transform,
                white_alpha_bg=config['data']['white_alpha_bg'])

    dset.H = dset.W = imsize
    dset.focal = W/2 * 1 / np.tan((.5 * fov * np.pi/180.))
    radius = config['data']['radius']
    if isinstance(radius, str):
        radius = tuple(float(r) for r in radius.split(','))
    dset.radius = radius
    print('Loaded {}'.format(dset_type), imsize, len(dset), config.root)
    return dset

def get_hwfr(config):
    H = W = config['data']['imsize']
    fov = config['data']['fov']
    focal = W/2 * 1 / np.tan((.5 * fov * np.pi/180.))
    radius = config['data']['radius']
    if isinstance(radius, str):
        radius = tuple(float(r) for r in radius.split(','))
    return [H,W,focal,radius]

def compute_render_poses(config):
    N = 40
    radius = config['data']['radius']
    render_radius = radius
    if isinstance(radius, str):
        radius = tuple(float(r) for r in radius.split(','))
        render_radius = max(radius)
    theta = 0.5 * (to_theta(config['data']['vmin']) + to_theta(config['data']['vmax']))
    angle_range = (to_phi(config['data']['umin']), to_phi(config['data']['umax']))
    return get_render_poses(render_radius, angle_range=angle_range, theta=theta, N=N)

def get_render_poses(radius, angle_range=(0, 360), theta=0, N=40, swap_angles=False):
    poses = []
    theta = max(0.1, theta)
    for angle in np.linspace(angle_range[0],angle_range[1],N+1)[:-1]:
        angle = max(0.1, angle)
        if swap_angles:
            loc = polar_to_cartesian(radius, theta, angle, deg=True)
        else:
            loc = polar_to_cartesian(radius, angle, theta, deg=True)
        R = look_at(loc)[0]
        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        poses.append(RT)
    return torch.from_numpy(np.stack(poses))


def build_models(config, disc=True):
    

    config_nerf = Namespace(**config['nerf'])
    # Update config for NERF
    config_nerf.chunk = min(config['train']['chunk'], 1024*config['train']['batch_size'])     # let batch size for training with patches limit the maximal memory
    config_nerf.netchunk = config['train']['netchunk']
    config_nerf.white_bkgd = config['data']['white_bkgd']
    config_nerf.feat_dim = config['train']['noise_dim']
    config_nerf.feat_dim_appearance = config['train']['noise_dim_appearance']

    render_kwargs_train, render_kwargs_test, params, named_parameters = create_nerf(config_nerf)

    bds_dict = {'near': config['data']['near'], 'far': config['data']['far']}
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    ray_sampler = FlexGridRaySampler(N_samples=config['ray_sampler']['N_samples'],
                                     min_scale=config['ray_sampler']['min_scale'],
                                     max_scale=config['ray_sampler']['max_scale'],
                                     scale_anneal=config['ray_sampler']['scale_anneal'],
                                     orthographic=config['data']['orthographic'])

    H, W, f, r = get_hwfr(config)
    generator = Generator(H, W, f, r,
                          ray_sampler=ray_sampler,
                          render_kwargs_train=render_kwargs_train, render_kwargs_test=render_kwargs_test,
                          parameters=params, named_parameters=named_parameters,
                          chunk=config_nerf.chunk,
                          range_u=(float(config['data']['umin']), float(config['data']['umax'])),
                          range_v=(float(config['data']['vmin']), float(config['data']['vmax'])),
                          orthographic=config['data']['orthographic'],
                          )

    discriminator = None
    if disc:
        disc_kwargs = {'nc': 3,       # channels for patch discriminator
                       'ndf': config['discriminator']['ndf'],
                       'imsize': int(np.sqrt(config['ray_sampler']['N_samples'])),
                       'hflip': config['discriminator']['hflip']}

        discriminator = Discriminator(**disc_kwargs)

    return generator, discriminator

def build_discriminator(config):
    disc_kwargs = {'nc': 3,       # channels for patch discriminator
                   'ndf': config['ndf'],
                   'imsize': int(np.sqrt(config['ray_sampler']['N_samples'])),
                   'hflip': config['hflip']}

    return Discriminator(**disc_kwargs)

def build_generator(config):
    generator, _ = build_models(config, disc=False)
    return generator

def build_lr_scheduler(optimizer, config, last_epoch=-1):
    import torch.optim as optim
    step_size = config['train']['lr_anneal_every']
    if isinstance(step_size, str):
        milestones = [int(m) for m in step_size.split(',')]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=config['train']['lr_anneal'],
            last_epoch=last_epoch)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=config['train']['lr_anneal'],
            last_epoch=last_epoch
        )
    return lr_scheduler
