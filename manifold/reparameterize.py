from typing import Dict, Tuple

import torch
from torch.distributions import gamma as gamma


def soft_threshold(z: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(torch.abs(z) - lambda_) * torch.sign(z)


def draw_noise_samples(shape: Tuple[int], device: torch.device):
    return torch.rand(shape, device=device) - 0.5


def reparameterize(
    logscale: torch.Tensor,
    shift: torch.Tensor,
    noise: torch.Tensor,
    threshold: float,
):
    # Laplacian reparameterize
    scale = torch.exp(logscale)
    eps = -scale * torch.sign(noise) * torch.log((1.0 - 2.0 * torch.abs(noise)).clamp(min=1e-6, max=1e6))
    c = shift + eps

    # We do this weird detaching pattern because in certain cases we want gradient to flow through lambda_
    # In the case where lambda_ is constant, this is the same as c_thresh.detach() in the final line.
    c_thresh = soft_threshold(eps.detach(), threshold)
    non_zero = torch.nonzero(c_thresh, as_tuple=True)
    c_thresh[non_zero] = (shift[non_zero].detach()) + c_thresh[non_zero]
    c = c + c_thresh - c.detach()
    return c


def compute_kl(
    enc_logscale: torch.Tensor,
    enc_shift: torch.Tensor,
    prior_logscale: torch.Tensor,
    prior_shift: torch.Tensor,
    detach_shift: bool = True,
):
    kl_loss = 0.0
    if detach_shift:
        enc_shift = enc_shift.detach()

    encoder_scale, prior_scale = torch.exp(enc_logscale), torch.exp(prior_logscale)
    laplace_kl = ((enc_shift - prior_shift).abs() / prior_scale) + prior_logscale - enc_logscale - 1
    laplace_kl += (encoder_scale / prior_scale) * (-((enc_shift - prior_shift).abs() / encoder_scale)).exp()
    kl_loss += laplace_kl.sum(dim=-1)

    return kl_loss
