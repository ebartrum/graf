import torch 
from torch.nn import functional as F
from torch import autograd

def graf(pl_module, batch, batch_idx, optimizer_idx):
    x_real = batch
    
    pl_module.generator.ray_sampler.iterations = pl_module.global_step//2   # for scale annealing

    # Sample patches for real data
    rgbs = pl_module.img_to_patch(x_real.to(pl_module.device))          # N_samples x C
    z = torch.randn(pl_module.cfg['train']['batch_size'],
            pl_module.cfg['train']['noise_dim'], device=pl_module.device)

    if optimizer_idx == 0:
        return graf_discriminator_trainstep(pl_module, rgbs,z=z)
    if optimizer_idx == 1:
        return graf_generator_trainstep(pl_module, z=z)

def graf_generator_trainstep(pl_module, z):
    if pl_module.cfg['nerf']['decrease_noise']:
      pl_module.generator.decrease_nerf_noise(pl_module.global_step//2)

    x_fake = pl_module.generator(z)
    d_fake = pl_module.discriminator(x_fake)
    gloss = compute_loss(d_fake, 1)
    pl_module.log('generator_loss', gloss)
    return gloss

def graf_discriminator_trainstep(pl_module, x_real, z):
    # On real data
    x_real.requires_grad_()
    d_real = pl_module.discriminator(x_real)
    dloss_real = compute_loss(d_real, 1)

    # On fake data
    with torch.no_grad():
        x_fake = pl_module.generator(z)

    x_fake.requires_grad_()
    d_fake = pl_module.discriminator(x_fake)
    dloss_fake = compute_loss(d_fake, 0)
    dloss = (dloss_real + dloss_fake)
    r1_reg = pl_module.reg_param * compute_grad2(d_real, x_real).mean()
    pl_module.log('discriminator_loss', dloss)
    pl_module.log('regularizer_loss', r1_reg)

    return r1_reg + dloss

def compute_loss(d_outs, target):
    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    loss = 0
    for d_out in d_outs:
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss += F.binary_cross_entropy_with_logits(d_out, targets)
    return loss / len(d_outs)

def compute_grad2(d_outs, x_in):
    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    reg = 0
    for d_out in d_outs:
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg += grad_dout2.view(batch_size, -1).sum(1)
    return reg / len(d_outs)
