import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ExperimentConfig
from eval.get_data import get_data
from eval.knn import eval_knn
from eval.sgd import eval_sgd
from manifold.transop import TransportOperator
from model import get_head, get_model


class BaseMethod(nn.Module):
    """
        Base class for self-supervised loss implementation.
        It includes encoder and head for training, evaluation function.
    """

    def __init__(self, cfg: ExperimentConfig, devices):
        super().__init__()
        self.model, self.out_size = get_model(cfg.ssl_cfg.arch, cfg.data_cfg.dataset, devices)
        self.head = get_head(self.out_size, cfg.ssl_cfg)
        self.knn = cfg.knn_num_neighbor
        self.num_pairs = cfg.data_cfg.num_samples * (cfg.data_cfg.num_samples - 1) // 2
        self.eval_head = cfg.eval_header
        self.emb_size = cfg.ssl_cfg.head_output_dim
        self.cfg = cfg

    def get_method_param_group(self):
        raise NotImplementedError

    def manifold_augmentation(self, z0, z1, manifold_operator):
        z1_hat, z0_tilde, z1_tilde, kl_loss, params = manifold_operator(z0, z1)
        transop_loss = F.mse_loss(z1_hat, z1)
        manifold_loss = transop_loss + self.cfg.manifold_cfg.kl_loss_weight*kl_loss

        if self.cfg.manifold_cfg.enable_shift_l2:
            shift_l2 = (params[1]**2).sum(dim=-1).mean()
            manifold_loss += self.cfg.manifold_cfg.shift_l2_weight * shift_l2
        if self.cfg.manifold_cfg.enable_eigreg:
            psi = manifold_operator.psi
            psi = psi.reshape(-1, psi.shape[-1], psi.shape[-1])
            psi_use = psi_use[torch.randperm(len(psi_use))[:10]]
            eig_loss = (torch.real(torch.linalg.eigvals(psi_use)) ** 2).sum()
            manifold_loss += self.cfg.manifold_cfg.eigreg_weight * eig_loss

        return z0_tilde, z1_tilde, manifold_loss

    def forward(self, samples, manifold_operator=None):
        raise NotImplementedError

    def get_acc(self, ds_clf, ds_test):
        self.eval()
        if self.eval_head:
            model = lambda x: self.head(self.model(x))
            out_size = self.emb_size
        else:
            model, out_size = self.model, self.out_size
        # torch.cuda.empty_cache()
        x_train, y_train = get_data(model, ds_clf, out_size, "cuda")
        x_test, y_test = get_data(model, ds_test, out_size, "cuda")

        acc_knn = eval_knn(x_train.detach().cpu(), y_train.detach().cpu(), x_test.detach().cpu(), y_test.detach().cpu(), self.knn)
        acc_linear = eval_sgd(x_train, y_train, x_test, y_test)
        del x_train, y_train, x_test, y_test
        self.train()
        return acc_knn, acc_linear

    def step(self, progress):
        pass
