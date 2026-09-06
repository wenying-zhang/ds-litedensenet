"""Training loop, inference and metric computation.

Model selection uses validation AUROC only; the test split is scored once per
model per seed and never influences training.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, f1_score, balanced_accuracy_score,
                             roc_curve)

import config as C
from stats import expected_calibration_error


def set_seed(seed, deterministic=False):
    """Seed the generators, and optionally remove run-to-run variation.

    With deterministic=False the run is fast but not bit-reproducible: cuDNN
    picks convolution algorithms by timing them, and several backward kernels
    accumulate in non-deterministic order. Repeating a command then shifts
    AUROC by about as much as the seed-to-seed spread.

    With deterministic=True the same command gives the same numbers on the same
    machine and software stack, at a cost of roughly 30 to 100 per cent in
    training time. CUBLAS_WORKSPACE_CONFIG must be set before the CUDA context
    is created, so for strict determinism set it in the environment rather than
    relying on the fallback here:

        CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_ablation.py --deterministic
    """
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        p = torch.softmax(model(x), dim=1)[:, 1]
        probs.append(p.cpu().numpy()); labels.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(labels)


def youden_threshold(y, prob):
    """Threshold maximising Youden's J, clipped to the probability range.

    sklearn sets thresholds[0] above every observed score (np.inf on current
    versions) so that the curve starts at (0, 0). If no threshold achieves a
    positive J -- a model at or below chance, or a degenerate split -- argmax
    returns that first entry and the function would report a "probability"
    outside [0, 1]. Clipping keeps the returned value interpretable; a model in
    that state is broken either way, and the metrics computed from it will say
    so.
    """
    fpr, tpr, thr = roc_curve(y, prob)
    return float(np.clip(thr[int(np.argmax(tpr - fpr))], 0.0, 1.0))


def compute_metrics(y, prob, thr=0.5):
    y = np.asarray(y).astype(int)
    pred = (prob >= thr).astype(int)
    out = {"thr": float(thr)}
    out["auroc"] = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    out["auprc"] = average_precision_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    out["acc"] = float((pred == y).mean())
    out["bacc"] = float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out["sens"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    out["spec"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    out["ece"] = expected_calibration_error(y, prob)
    return out


def train_model(model, train_loader, val_loader, device,
                epochs=C.EPOCHS, lr=C.LR, weight_decay=C.WEIGHT_DECAY,
                patience=C.EARLY_PATIENCE, verbose=True):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = nn.CrossEntropyLoss()

    best_auc, best_state, wait = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
        sched.step()

        vprob, vy = predict(model, val_loader, device)
        vauc = roc_auc_score(vy, vprob) if len(np.unique(vy)) > 1 else 0.0
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  val_auroc={vauc:.4f}")
        if vauc > best_auc:
            best_auc, wait = vauc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  early stop @ epoch {ep} (best val_auroc={best_auc:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def load_pretrained_encoder(model, ckpt_path, device):
    """Load self-supervised pretrained weights, skipping the classifier head."""
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    msd = model.state_dict()
    # Exclude every head this repository builds. DenseNetVariant and
    # MobileNetV2 name theirs "classifier"; torchvision's ShuffleNetV2 names
    # its head "fc", and without the second test a pretrained ShuffleNetV2
    # checkpoint of the same class count would carry a fitted classifier into
    # a run reported as encoder-only pretraining.
    _HEADS = ("classifier", "fc")
    keep = {k: v for k, v in sd.items()
            if k in msd and not any(h in k.split(".") for h in _HEADS)
            and v.shape == msd[k].shape}
    msd.update(keep); model.load_state_dict(msd)
    print(f"  loaded {len(keep)} pretrained tensors from {ckpt_path}")
    return model
