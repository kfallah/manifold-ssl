import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import wandb

log = logging.getLogger(__name__)

def sweep_psi_path_plot(psi: torch.tensor, z0: np.array, c_mag: int) -> Figure:
    z = torch.tensor(z0).float().to(psi.device)[:psi.shape[-1]]

    # z = model.backbone(x_gpu[0])[0]
    # z = torch.tensor(z0[0][0]).to(default_device)
    # psi = model.contrastive_header.transop_header.transop.psi
    psi_norm = (psi.reshape(len(psi), -1) ** 2).sum(dim=-1)
    psi_idx = torch.argsort(psi_norm)
    latent_dim = len(z)

    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.4, top=0.9)

    for i in range(ax.size):
        row = int(i / 3)
        column = int(i % 3)
        curr_psi = psi_idx[-(i + 1)]

        coeff = torch.linspace(-c_mag, c_mag, 30, device=psi.device)
        T = torch.matrix_exp(coeff[:, None, None] * psi[None, curr_psi])
        z1_hat = (T @ z).squeeze(dim=-1)

        for z_dim in range(latent_dim):
            ax[row, column].plot(
                np.linspace(-c_mag, c_mag, 30),
                z1_hat[:, z_dim].detach().cpu().numpy(),
            )
        ax[row, column].title.set_text(
            f"Psi {curr_psi} - F-norm: {psi_norm[curr_psi]:.2E}"
        )

    return fig


def transop_plots(
    coefficients: np.array, psi: torch.tensor, z0: np.array
) -> Dict[str, Figure]:
    psi_norms = ((psi.reshape(len(psi), -1)) ** 2).sum(dim=-1).detach().cpu().numpy()
    count_nz = np.zeros(len(psi) + 1, dtype=int)
    total_nz = np.count_nonzero(coefficients, axis=1)
    for z in range(len(total_nz)):
        count_nz[total_nz[z]] += 1
    number_operator_uses = np.count_nonzero(coefficients, axis=0) / len(coefficients)

    psi_mag_fig = plt.figure(figsize=(20, 4))
    plt.bar(np.arange(len(psi)), psi_norms, width=1)
    plt.xlabel("Transport Operator Index", fontsize=18)
    plt.ylabel("F-Norm", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("F-Norm of Transport Operators", fontsize=20)

    coeff_use_fig = plt.figure(figsize=(20, 4))
    plt.bar(np.arange(len(psi) + 1), count_nz, width=1)
    plt.xlabel("Number of Coefficients Used per Point Pair", fontsize=18)
    plt.ylabel("Occurences", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Number of Non-Zero Coefficients", fontsize=20)

    psi_use_fig = plt.figure(figsize=(20, 4))
    plt.bar(np.arange(len(psi)), number_operator_uses, width=1)
    plt.xlabel("Percentage of Point Pairs an Operator is Used For", fontsize=18)
    plt.ylabel("% Of Point Pairs", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Transport Operator Index", fontsize=20)

    psi_eig_plt = plt.figure(figsize=(8, 8))
    L = torch.linalg.eigvals(psi.detach())
    plt.scatter(
        torch.real(L).detach().cpu().numpy(), torch.imag(L).detach().cpu().numpy()
    )
    plt.xlabel("Real Components of Eigenvalues", fontsize=18)
    plt.ylabel("Imag Components of Eigenvalues", fontsize=18)

    psi_sweep_1c_fig = sweep_psi_path_plot(psi.detach(), z0, 1)
    psi_sweep_5c_fig = sweep_psi_path_plot(psi.detach(), z0, 5)

    figure_dict = {
        "psi_mag_iter": psi_mag_fig,
        "coeff_use_iter": coeff_use_fig,
        "psi_use_iter": psi_use_fig,
        "psi_eig_plt": psi_eig_plt,
        "psi_sweep_1c": psi_sweep_1c_fig,
        "psi_sweep_5c": psi_sweep_5c_fig,
    }

    return figure_dict

def log_train_metrics():
    pass

def log_manifold_metrics(psi, c, z0, z1, z1_hat):
    coeff_nz = np.count_nonzero(c, axis=0)
    nz_tot = np.count_nonzero(coeff_nz)
    total_nz = np.count_nonzero(c, axis=1)
    avg_feat_norm = torch.linalg.norm(z0, axis=-1).mean()
    dist_bw_point_pairs = F.mse_loss(z1, z0).item()
    transop_dist = (
        F.mse_loss(
            z1,
            z1_hat,
            reduction="none",
        ).sum(dim=-1)
        / (
            F.mse_loss(
                z1,
                z0,
                reduction="none",
            ).sum(dim=-1)
            + 1e-6
        )
    ).mean(dim=-1)
    mean_dist_improvement = transop_dist.mean().item()

    psi_mag = torch.norm(psi.data.reshape(len(psi.data), -1), dim=-1)
    to_metrics = {
        "transop/avg_transop_mag": psi_mag.mean(),
        "transop/total_transop_used": nz_tot,
        "transop/avg_transop_used": total_nz.mean(),
        "transop/avg_coeff_mag": np.abs(c[np.abs(c) > 0]).mean(),
        "transop/avg_feat_norm": avg_feat_norm,
        "transop/dist_bw_point_pairs": dist_bw_point_pairs,
        "transop/mean_dist_improvement": mean_dist_improvement,
    }
    # if self.cfg.enable_console_logging:
    #     log.info(
    #         f"[TO iter {curr_iter}]:"
    #         + f", dist improve: {mean_dist_improvement:.3E}"
    #         + f", avg # to used: {total_nz.mean():.2f}/{len(psi)}"
    #         + f", avg coeff mag: {to_metrics['transop/avg_coeff_mag']:.3f}"
    #         + f", dist bw pp: {dist_bw_point_pairs:.3E}"
    #         + f", average to mag: {psi_mag.mean():.3E}"
    #         + f", avg feat norm: {avg_feat_norm:.2E}"
    #     )
    #     if self.model.model_cfg.header_cfg.transop_header_cfg.enable_variational_inference:
    #         distr_data = model_output.header_output.distribution_data
    #         scale = torch.exp(distr_data.encoder_params["logscale"])
    #         shift = distr_data.encoder_params["shift"]
    #         log.info(
    #             f"[Encoder params]: "
    #             + f"min scale: {scale.abs().min():.2E}"
    #             + f", max scale: {scale.abs().max():.2E}"
    #             + f", mean scale: {scale.mean():.2E}"
    #             + f", min shift: {shift.abs().min():.2E}"
    #             + f", max shift: {shift.abs().max():.2E}"
    #             + f", mean shift: {shift.abs().mean():.2E}"
    #         )
    #         if self.model.model_cfg.header_cfg.transop_header_cfg.vi_cfg.enable_learned_prior:
    #             prior_scale = torch.exp(distr_data.prior_params["logscale"])
    #             prior_shift = distr_data.prior_params["shift"]
    #             log.info(
    #                 f"[Prior params]: "
    #                 + f"min scale: {prior_scale.abs().min():.3E}"
    #                 + f", max scale: {prior_scale.abs().max():.3E}"
    #                 + f", mean scale: {prior_scale.mean():.3E}"
    #                 + f", min shift: {prior_shift.abs().min():.3E}"
    #                 + f", max shift: {prior_shift.abs().max():.3E}"
    #                 + f", mean shift: {prior_shift.abs().mean():.3E}"
    #             )

    # Generate transport operator plots
    fig_dict = transop_plots(c, psi, z0[0])
    for fig_name in fig_dict.keys():
        wandb.log({"transop_plt/" + fig_name: wandb.Image(fig_dict[fig_name])})
        plt.close(fig_dict[fig_name])