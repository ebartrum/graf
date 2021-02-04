import torch 

def graf(pl_module, batch, batch_idx, optimizer_idx):
    x_real = batch
    
    pl_module.generator.ray_sampler.iterations = pl_module.global_step   # for scale annealing

    # Sample patches for real data
    rgbs = pl_module.img_to_patch(x_real.to(pl_module.device))          # N_samples x C

    # Discriminator updates
    z = torch.randn(pl_module.cfg['training']['batch_size'], pl_module.cfg['z_dist']['dim'])
    dloss, reg = pl_module.gan_trainer.discriminator_trainstep(rgbs,z=z)
    pl_module.log('discriminator_loss', dloss)
    pl_module.log('regularizer_loss', reg)

    # Generators updates
    if pl_module.cfg['nerf']['decrease_noise']:
      pl_module.generator.decrease_nerf_noise(pl_module.global_step)

    z = torch.randn(pl_module.cfg['training']['batch_size'], pl_module.cfg['z_dist']['dim'])
    gloss = pl_module.gan_trainer.generator_trainstep(z=z)
    pl_module.log('generator_loss', gloss)

    # Update learning rate
    pl_module.g_scheduler.step()
    pl_module.d_scheduler.step()

    d_lr = pl_module.d_optimizer.param_groups[0]['lr']
    g_lr = pl_module.g_optimizer.param_groups[0]['lr']

    pl_module.log('discriminator_lr', d_lr)
    pl_module.log('generator_lr', g_lr)
