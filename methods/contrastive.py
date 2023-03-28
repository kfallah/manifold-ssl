from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from manifold.transop import TransportOperator

from .base import BaseMethod


def contrastive_loss(x0, x1, tau, norm):
    # https://github.com/google-research/simclr/blob/master/objective.py
    bsize = x0.shape[0]
    target = torch.arange(bsize, device=x0.device)
    eye_mask = torch.eye(bsize, device=x0.device) * 1e9
    if norm:
        x0 = F.normalize(x0, p=2, dim=1)
        x1 = F.normalize(x1, p=2, dim=1)
    logits00 = x0 @ x0.t() / tau - eye_mask
    logits11 = x1 @ x1.t() / tau - eye_mask
    logits01 = x0 @ x1.t() / tau
    logits10 = x1 @ x0.t() / tau
    return (
        F.cross_entropy(torch.cat([logits01, logits00], dim=1), target)
        + F.cross_entropy(torch.cat([logits10, logits11], dim=1), target)
    ) / 2


class Contrastive(BaseMethod):
    """ implements contrastive loss https://arxiv.org/abs/2002.05709 """

    def __init__(self, cfg, devices):
        """ init additional BN used after head """
        super().__init__(cfg, devices)
        self.bn_last = nn.BatchNorm1d(cfg.ssl_cfg.head_output_dim)
        self.loss_f = partial(contrastive_loss, tau=cfg.ssl_cfg.tau, norm=cfg.ssl_cfg.norm_latent)

    def forward(self, samples, manifold_operator=None):
        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True)) for x in samples]

        manifold_loss = 0.
        if manifold_operator is not None:
            z0, z1 = h[0], h[1]
            z0_tilde, z1_tilde, manifold_loss = self.manifold_augmentation(z0, z1, manifold_operator)
            h = [z0_tilde, z1_tilde]

        h = self.bn_last(self.head(torch.cat(h)))
        loss = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                x0 = h[i * bs : (i + 1) * bs]
                x1 = h[j * bs : (j + 1) * bs]
                loss += self.loss_f(x0, x1)
        loss /= self.num_pairs
        return loss + manifold_loss
