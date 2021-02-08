import torch
from .metrics import inception_score

class Evaluator(object):
    def __init__(self, generator, noise_dim, batch_size=64,
                 inception_nsamples=60000, device=None):
        self.generator = generator
        self.inception_nsamples = inception_nsamples
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.device = device

    def compute_inception_score(self):
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
            ztest = torch.randn(self.batch_size, cfg['z_dist']['dim'])

            samples = self.generator(ztest)
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        imgs = imgs[:self.inception_nsamples]
        score, score_std = inception_score(
            imgs, device=self.device, resize=True, splits=10
        )

        return score, score_std

    def create_samples(self, z):
        self.generator.eval()
        batch_size = z.size(0)
        # Sample x
        with torch.no_grad():
            x = self.generator(z)
        return x
