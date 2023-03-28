import torch
import torch.nn as nn

from config import ManifoldConfig
from manifold.vi_encoder import CoefficientEncoder


class TransportOperator(nn.Module):
    """
        Transport operator class for applying manifold augmentations.
    """

    def __init__(self, cfg: ManifoldConfig, feature_dim: int):
        super().__init__()
        block_count, op_dim = 1, feature_dim
        if cfg.enable_block:
            block_count = feature_dim // cfg.block_dim
            op_dim = cfg.block_dim

        if cfg.enable_stable_init:
            self.psi = nn.Parameter(torch.zeros((cfg.dictionary_size, block_count, op_dim, op_dim)), requires_grad=True)
            real = (torch.rand((cfg.dictionary_size, block_count)) - 0.5) * cfg.real_range_init
            imag = (torch.rand((cfg.dictionary_size, block_count)) - 0.5) * cfg.image_range_init
            for i in range(0, op_dim, 2):
                self.psi.data[..., i, i] = real
                self.psi.data[..., i + 1, i] = imag
                self.psi.data[..., i, i + 1] = -imag
                self.psi.data[..., i + 1, i + 1] = real
        else:
            self.psi = nn.Parameter(torch.mul(torch.randn((cfg.dictionary_size, block_count, op_dim, op_dim)), 1.0e-3), requires_grad=True)

        self.cfg = cfg
        self.coeff_enc = CoefficientEncoder(cfg, op_dim)
        self.op_dim = op_dim
        self.block_count = block_count

    def forward(self, z0, z1):
        c_enc, c0_prior, c1_prior, kl_loss, params = self.coeff_enc(z0.detach(), z1.detach(), self.psi.detach())

        A_transform = torch.einsum("bm,smuv->bsuv", c_enc, self.psi)
        T_transform = torch.matrix_exp(A_transform.reshape(-1, self.op_dim, self.op_dim)).reshape(*A_transform.shape)
        z1_hat = (T_transform @ z0.reshape(len(z0), self.block_count, -1, 1)).reshape(len(z0), -1)

        A0 = torch.einsum("bm,smuv->bsuv", c0_prior, self.psi)
        T0 = torch.matrix_exp(A0.reshape(-1, self.op_dim, self.op_dim)).reshape(*A0.shape)
        z0_tilde = (T0 @ z0.reshape(len(z0), self.block_count, -1, 1)).reshape(len(z0), -1)
        A1 = torch.einsum("bm,smuv->bsuv", c1_prior, self.psi)
        T1 = torch.matrix_exp(A1.reshape(-1, self.op_dim, self.op_dim)).reshape(*A1.shape)
        z1_tilde = (T1 @ z1.reshape(len(z1), self.block_count, -1, 1)).reshape(len(z1), -1)

        return z1_hat, z0_tilde, z1_tilde, kl_loss, params