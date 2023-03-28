import logging
import warnings

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm, trange

import wandb
from config import ExperimentConfig
from datasets import get_ds
from methods import get_method
from metric_log import log_train_metrics

warnings.filterwarnings("ignore")

def register_configs() -> None:
    cs.store(name="exp_cfg", node=ExperimentConfig)

def get_scheduler(optimizer, cfg: ExperimentConfig):
    if cfg.lr_scheduler == "cos":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg.epoch,
            eta_min=cfg.lr_min,
        )
    elif cfg.lr_scheduler == "step":
        m = [cfg.epoch - a for a in cfg.lr_drop]
        return MultiStepLR(optimizer, milestones=m, gamma=cfg.lr_drop_gamma)
    else:
        return None


@hydra.main(version_base=None, config_path="config", config_name="base")
def run_experiment(cfg: ExperimentConfig):
    wandb.init(
        project=cfg.wandb_project, 
        config=cfg, 
        mode="online" if cfg.enable_wandb else "disabled",
        settings=wandb.Settings(start_method="thread"),
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    default_device = cfg.devices[0]

    ds = get_ds(cfg.data_cfg.dataset)(cfg.data_cfg.batch_size, cfg.data_cfg, cfg.data_cfg.num_workers)
    model = get_method(cfg.ssl_cfg.ssl_method)(cfg, cfg.devices)
    model.to(default_device).train()
    if len(cfg.load_filename) > 0:
        model.load_state_dict(torch.load(cfg.load_filename))

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(optimizer, cfg)

    # Set random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    eval_every = cfg.eval_freq
    lr_warmup = 0 if cfg.lr_warmup else 500
    cudnn.benchmark = True

    for ep in trange(cfg.epoch, position=0):
        loss_ep = []
        iters = len(ds.train)
        for n_iter, (samples, _) in enumerate(tqdm(ds.train, position=1)):
            if lr_warmup < 500:
                lr_scale = (lr_warmup + 1) / 500
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.lr * lr_scale
                lr_warmup += 1

            optimizer.zero_grad()
            loss = model(samples)
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())
            model.step(ep / cfg.epoch)
            if cfg.lr_scheduler == "cos" and lr_warmup >= 500:
                scheduler.step()

        if cfg.lr_scheduler == "step":
            scheduler.step()

        if (ep + 1) % eval_every == 0:
            acc_knn, acc = model.get_acc(ds.clf, ds.test)
            wandb.log({"eval/acc": acc[1], "eval/acc_5": acc[5], "eval/acc_knn": acc_knn}, commit=False)

        if (ep + 1) % 100 == 0:
            fname = f"checkpoints/{cfg.ssl_cfg.method}_{cfg.data_cfg.dataset}_{ep}.pt"
            torch.save(model.state_dict(), fname)

        log_train_metrics()
        wandb.log({"loss": np.mean(loss_ep), "ep": ep})

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    cs = ConfigStore.instance()
    register_configs()
    run_experiment()