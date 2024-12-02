import numpy as np
import torch
from functools import partial
import torch.nn.functional as F
import torchvision

from utils import diff2clf, clf2diff, normalize, resize

def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas).float()

# our method
class PurificationForward(torch.nn.Module):
    def __init__(self, clf, diffusion, max_timestep, attack_steps, sampling_method, guidance, device):
        super().__init__()
        self.clf = clf
        self.diffusion = diffusion
        self.betas = get_beta_schedule(1e-4, 2e-2, 1000).to(device)
        self.max_timestep = max_timestep
        self.attack_steps = attack_steps
        self.sampling_method = sampling_method
        self.guidance = guidance
        assert sampling_method in ['ddim', 'ddpm', 'random']
        if self.sampling_method == 'ddim':
            self.eta1 = 0
            self.eta2 = 1
        elif self.sampling_method == 'ddpm':
            self.eta1 = 1
            self.eta2 = 1
        elif self.sampling_method == 'random':
            self.eta1 = 1
            self.eta2 = 0

    def compute_alpha(self, t):
        beta = torch.cat(
            [torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def get_noised_x(self, x, t):
        e = torch.randn_like(x)
        if type(t) == int:
            t = (torch.ones(x.shape[0]) * t).to(x.device).long()
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()
        return x

# main algorithm
    def denoising_process(self, x, seq, ref, scale=30000):
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        ori_x = ref
        x_t = x
        count = 1
        k = 1
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            alpha_t = self.compute_alpha(t.long())
            alpha_next_t = self.compute_alpha(next_t.long())

            # estimate mediator x_0t
            s_hat = self.diffusion(x_t, t)
            x_0t = (x_t - s_hat * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

            # mediator guidance
            guidances = 0.
            if (count % k == 0) & self.guidance:
                x_t.requires_grad_()
                with torch.enable_grad():
                    x_0t = (x_t - s_hat * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
                    dist = torch.nn.MSELoss()(x_0t, ori_x)

                gradient = torch.autograd.grad(dist, [x_0t], retain_graph=True)[0].detach()
                Rt = scale * alpha_t.sqrt()
                guidances = Rt * gradient
            x_0t = x_0t - guidances

            # update x_t (next)
            z = torch.randn_like(x)
            coeff1 = self.eta1 * ((1 - alpha_t / alpha_next_t) * (1 - alpha_next_t) / (1 - alpha_t)).sqrt()
            coeff2 = self.eta2 * ((1 - alpha_next_t) - coeff1 ** 2).sqrt()
            x_t = alpha_next_t.sqrt() * x_0t + coeff1 * z + coeff2 * s_hat
            count += 1
        return x_t

    def classify(self, x):
        logits = self.clf(x)
        return logits

    def forward(self, x):
        # diffusion part
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(noised_x, self.attack_steps[i], ref=x_diff)

        # classifier part
        x_clf = diff2clf(x_diff)
        logits = self.clf(x_clf)
        return logits
