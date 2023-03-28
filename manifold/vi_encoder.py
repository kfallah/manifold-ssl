import torch
import torch.nn as nn

from config import ManifoldConfig
from manifold.reparameterize import (compute_kl, draw_noise_samples,
                                     reparameterize)


class CoefficientEncoder(nn.Module):
    """
        Base class for self-supervised loss implementation.
        It includes encoder and head for training, evaluation function.
    """

    def __init__(self, cfg: ManifoldConfig, input_dim: int):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(2*input_dim, cfg.feature_dim*4),
            nn.LeakyReLU(),
            nn.Linear(cfg.feature_dim*4, cfg.feature_dim*4),
            nn.LeakyReLU(),
            nn.Linear(cfg.feature_dim*4, cfg.feature_dim),
        )
        self.enc_scale = nn.Linear(cfg.feature_dim, cfg.dictionary_size)
        self.enc_shift = nn.Linear(cfg.feature_dim, cfg.dictionary_size)

        if cfg.learn_prior:
            self.prior = nn.Sequential(
                nn.Linear(input_dim, cfg.feature_dim*4),
                nn.LeakyReLU(),
                nn.Linear(cfg.feature_dim*4, cfg.feature_dim*4),
                nn.LeakyReLU(),
                nn.Linear(cfg.feature_dim*4, cfg.feature_dim),
            )
            self.prior_scale = nn.Linear(cfg.feature_dim, cfg.dictionary_size)

    def max_elbo(self, z0, z1, psi, enc_logscale, enc_shift):
        with torch.no_grad():
            noise_list = []
            loss_list = []
            l1_list = []
            enc_logscale = enc_logscale.unsqueeze(0).repeat(self.vi_cfg.samples_per_iter, 1, 1)
            for _ in range(self.vi_cfg.total_num_samples // self.vi_cfg.samples_per_iter):
                noise = draw_noise_samples(enc_logscale.shape, enc_logscale.device)
                # s x b x m
                c = reparameterize(enc_logscale, enc_shift, noise)
                # Handle the case where we have a dictionary per block
                T = torch.einsum("slbm,lmuv->sbluv", c, psi)
                s, b, l, n, _ = T.shape
                # s x b x l x d x d
                T = torch.matrix_exp(T.reshape(-1, n, n)).reshape(s, b, l, n ,n)
                # b x l x d
                z0_block = z0.reshape(len(z0), l, n)
                # s x b x l x d
                z1_hat_block = (T @ z0_block.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                # s x b x D
                z1_hat = z1_hat_block.reshape(s, len(x0), -1)
                # s x b
                transop_loss = torch.nn.functional.mse_loss(
                    z1_hat, z1.unsqueeze(0).expand(s, -1, -1), reduction="none"
                ).mean(dim=-1)

                noise_list.append(noise.squeeze(1))
                loss_list.append(transop_loss)

            # S x b x m
            noise_list = torch.cat(noise_list, dim=0)
            # S x b
            loss_list = torch.cat(loss_list, dim=0)
            # S x b
            max_elbo = torch.argmin(loss_list, dim=0).detach()
            optimal_noise = noise_list[max_elbo, torch.arange(len(max_elbo))]
            return optimal_noise

    def forward(self, z0, z1, psi):
        z_feat = self.encoder(torch.cat([z0,z1], dim=-1))
        enc_logscale, enc_shift = self.enc_scale(z_feat), self.enc_shift(z_feat)

        if self.cfg.enable_max_sample:
            noise = self.max_elbo(z0, z1, psi, enc_logscale, enc_shift)
        else:
            noise = draw_noise_samples(enc_logscale.shape, enc_logscale.device)
        c_enc = reparameterize(enc_logscale, enc_shift, noise)

        if self.cfg.learn_prior:
            z0_feat = self.prior(z0)
            z0_logscale = self.prior_scale(z0_feat)
            z1_feat = self.prior(z1)
            z1_logscale = self.prior_scale(z1_feat)
        else: 
            z0_logscale = torch.log(torch.ones_like(enc_logscale) * self.cfg.scale_prior)
            z1_logscale = torch.log(torch.ones_like(enc_logscale) * self.cfg.scale_prior)

        z0_shift = torch.ones_like(enc_shift) * self.cfg.shift_prior
        z1_shift = torch.ones_like(enc_shift) * self.cfg.shift_prior
        c0_prior = reparameterize(z0_logscale, z0_shift, noise)
        c1_prior = reparameterize(z1_logscale, z1_shift, noise)

        kl_loss = compute_kl(enc_logscale, enc_shift, z0_logscale, z0_shift)

        return c_enc, c0_prior, c1_prior, kl_loss, (enc_logscale, enc_shift, z0_logscale, z0_shift, z1_logscale, z1_shift)

