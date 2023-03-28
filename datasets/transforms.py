import torchvision.transforms as T

from config import DataConfig


def aug_transform(crop, base_transform, cfg: DataConfig, extra_t=[]):
    """ augmentation transform generated from config """
    return T.Compose(
        [
            T.RandomApply(
                [T.ColorJitter(cfg.cj_brightness, cfg.cj_contrast, cfg.cj_saturation, cfg.cj_hue)], p=cfg.cj_prob
            ),
            T.RandomGrayscale(p=cfg.grayscale_prob),
            T.RandomResizedCrop(
                crop,
                scale=(cfg.min_crop, cfg.max_crop),
                ratio=(cfg.min_crop_ratio, cfg.max_crop_ratio),
                interpolation=3,
            ),
            T.RandomHorizontalFlip(p=cfg.hf_prob),
            *extra_t,
            base_transform(),
        ]
    )


class MultiSample:
    """ generates n samples with augmentation """

    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))
