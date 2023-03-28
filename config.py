from dataclasses import dataclass
from typing import Tuple


@dataclass
class SSLConfig:
    ssl_method: str = "contrastive"
    norm_latent: bool = True
    add_bn: bool = True
    head_layers: int = 2
    head_hidden_dim: int = 1024
    head_output_dim: int = 64
    arch: str = "resnet18"

    # For InfoNCE loss
    tau: float = 0.5

    # Config for BYOL
    byol_tau: float = 0.99

    # Config for WMSE
    w_eps: float = 0.0
    w_iter: int = 1

@dataclass
class DataConfig:
    dataset: str = "cifar10"
    batch_size: int = 1024
    num_workers: int = 32
    num_samples: int = 2

    cj_brightness: float = 0.4
    cj_contrast: float = 0.4
    cj_saturation: float = 0.4
    cj_hue: float = 0.1
    cj_prob: float = 0.8
    grayscale_prob: float = 0.1
    min_crop: float = 0.2
    max_crop: float = 1.0
    min_crop_ratio: float = 0.75
    max_crop_ratio: float = 4/3
    hf_prob: float = 0.5

@dataclass
class ManifoldConfig:
    enable_manifold_aug: bool = False
    dictionary_size: int = 100
    
    batch_size: int = 128
    lr: float = 1.0e-3
    wd: float = 1.0e-5
    vi_lr: float = 1.0e-4
    vi_wd: float = 1.0e-6

    enable_stable_init: bool = True
    real_range_init: float = 0.0001
    image_range_init: float = 6.0

    enable_max_sample: bool = True
    sample_per_iter: int = 50
    total_samples: int = 100

    enable_block: bool = True
    block_dim: int = 64

    # VI Config
    feature_dim: int = 512
    threshold: float = 0.01
    scale_prior: float = 0.05
    shift_prior: float = 0.00
    learn_prior: bool = True
    prior_shift: bool = False

    # Loss config
    transop_loss_weight: float = 1.0
    kl_loss_weight: float = 1.0e-5
    enable_shift_l2: bool = True
    shift_l2_weight: bool = 2.0e-3
    enable_eigreg: bool = True
    eigreg_weight: bool = 1.0e-6
       

@dataclass
class ExperimentConfig:
    # Hierarchical configurations used for experiment
    data_cfg: DataConfig = DataConfig()
    ssl_cfg: SSLConfig = SSLConfig()
    manifold_cfg: ManifoldConfig = ManifoldConfig()

    # Experimental metadata
    exp_name: str = "SSL_run"
    exp_dir: str = "results"
    devices: Tuple[int] = (0,)
    enable_wandb: bool = True
    wandb_project: str = "manifold-ssl"
    load_filename: str = ""
    seed: int = 0
    epoch: int = 1000

    # Initial lr
    lr: float = 3.0e-3
    lr_min: float = 1.0e-5
    # Experimental settings
    lr_warmup: bool = True
    # Options: cos, step, none
    lr_scheduler: str = "cos"
    lr_drop: Tuple[int] = (50, 25)
    lr_drop_gamma: float = 0.2
    # weight decay
    weight_decay: float = 1.0e-6

    # Eval config
    eval_freq: int = 100
    # Whether to use project head for eval
    eval_header: bool = False
    knn_num_neighbor: int = 5
