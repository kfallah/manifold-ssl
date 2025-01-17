import torch

from config import get_cfg
from datasets import get_ds
from eval.get_data import get_data
from eval.knn import eval_knn
from eval.lbfgs import eval_lbfgs
from eval.sgd import eval_sgd
from methods import get_method

if __name__ == "__main__":
    cfg = get_cfg()

    model_full = get_method(cfg.method)(cfg)
    model_full.cuda().eval()
    if cfg.fname is None:
        print("evaluating random model")
    else:
        model_full.load_state_dict(torch.load(cfg.fname))

    ds = get_ds(cfg.dataset)(None, cfg, cfg.num_workers)
    device = "cpu" if cfg.clf == "lbfgs" else "cuda"
    if cfg.eval_head:
        model = lambda x: model_full.head(model_full.model(x))
        out_size = cfg.emb
    else:
        model = model_full.model
        out_size = model_full.out_size
    x_train, y_train = get_data(model, ds.clf, out_size, device)
    x_test, y_test = get_data(model, ds.test, out_size, device)

    acc = eval_sgd(x_train, y_train, x_test, y_test)
    print(acc)
    acc = eval_knn(x_train, y_train, x_test, y_test)
    print(acc)
    acc = eval_lbfgs(x_train, y_train, x_test, y_test)
    print(acc)
