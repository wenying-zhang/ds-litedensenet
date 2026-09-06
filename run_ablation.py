"""Architecture ablation and comparison against lightweight baselines.

For each model and seed: train on the patient-wise train split, select the best
epoch by validation AUROC, then score the locked test split once. Results are
reported as mean and standard deviation over seeds, with bootstrap 95 per cent
intervals and significance against DS-LiteDenseNet (paired t-test over seeds,
plus a DeLong test on one held seed).

Usage:
    python run_ablation.py --task pneumonia
    python run_ablation.py --task pneumonia --pretrained checkpoints/ssl.pth
    python run_ablation.py --task pneumonia --label_fraction 0.1
"""
import os
import sys

# CUBLAS_WORKSPACE_CONFIG has to be in the environment before the CUDA context is
# created, which happens on the first torch call -- setting it later is silently
# ineffective. Doing it here, above `import torch`, means `--deterministic` works
# even when the shell variable was forgotten, and removes a step that is easy to
# get wrong (PowerShell's `set` is not cmd's `set`, and a new terminal tab does
# not inherit it).
if "--deterministic" in sys.argv:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config as C
from datasets import CXRDataset, make_weighted_sampler, seed_worker
from models import build_model, count_params, count_flops_g
from engine import set_seed, train_model, predict, compute_metrics, youden_threshold, load_pretrained_encoder
from stats import delong_roc_test, paired_ttest, noninferiority_test


def get_loaders(task, seed, label_fraction=1.0, deterministic=False):
    man = C.MANIFEST_DIR / f"{task}_internal.csv"
    df = pd.read_csv(man)
    tr = df[df.split == "train"].reset_index(drop=True)
    if label_fraction < 1.0:
        # Stratified subsample of the training split, drawn within each label
        # group so the class ratio is preserved at every fraction.
        tr = (tr.groupby("label", group_keys=False)
                .sample(frac=label_fraction, random_state=seed)
                .reset_index(drop=True))
    tr_ds = CXRDataset(tr, train=True)
    va_ds = CXRDataset(df[df.split == "val"], train=False)
    te_ds = CXRDataset(df[df.split == "test"], train=False)
    # A seeded generator makes the sampling order and the worker seeds
    # reproducible; without it the draw order varies between runs.
    gen = torch.Generator().manual_seed(seed)
    sampler = make_weighted_sampler(tr_ds.labels(), generator=gen)
    loader_kw = dict(generator=gen, worker_init_fn=seed_worker) if deterministic else {}
    # Keep worker processes alive across epochs. Under the spawn start method
    # each worker re-imports torch, cv2 and pydicom, which dominates runtime on
    # small corpora such as TB (about 15 batches per epoch). Throughput only:
    # data, ordering and results are unaffected.
    _pw = C.NUM_WORKERS > 0
    # A final training batch of exactly one sample makes nn.BatchNorm2d raise,
    # because it cannot compute a variance, and the run dies after the epochs
    # already spent. Dropping that batch is the standard guard, but dropping it
    # unconditionally would discard up to BATCH_SIZE-1 samples per epoch from
    # every run. None of the configurations reported in the manuscript leaves a
    # remainder of one, so the guard is applied only when it is needed and every
    # reported run is unaffected.
    _drop_last = (len(tr_ds) % C.BATCH_SIZE) == 1
    if _drop_last:
        print(f"  [loader] {len(tr_ds)} training images leaves a final batch of "
              f"1 at batch size {C.BATCH_SIZE}; dropping it (BatchNorm needs "
              f"more than one sample per channel).")
    tr_ld = DataLoader(tr_ds, batch_size=C.BATCH_SIZE, sampler=sampler,
                       num_workers=C.NUM_WORKERS, pin_memory=True,
                       persistent_workers=_pw, drop_last=_drop_last, **loader_kw)
    va_ld = DataLoader(va_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                       num_workers=C.NUM_WORKERS, pin_memory=True,
                       persistent_workers=_pw, worker_init_fn=seed_worker)
    te_ld = DataLoader(te_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                       num_workers=C.NUM_WORKERS, pin_memory=True,
                       persistent_workers=_pw, worker_init_fn=seed_worker)
    return tr_ld, va_ld, te_ld


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pneumonia",
                    choices=["pneumonia", "tb", "covid"])
    ap.add_argument("--pretrained", default=None, help="SSL encoder checkpoint")
    ap.add_argument("--label_fraction", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--models", nargs="+", default=None,
                    help="subset of models to run (default: C.ABLATION_MODELS)")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS,
                    help="override the training epoch budget")
    ap.add_argument("--ni_margin", type=float, default=0.02,
                    help="AUROC margin for the non-inferiority test (default 0.02)")
    ap.add_argument("--deterministic", action="store_true",
                    help="make runs bit-reproducible on one machine; costs "
                         "roughly 30-100%% in training time")
    args = ap.parse_args()

    models_to_run = args.models if args.models else C.ABLATION_MODELS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{args.task}_lf{args.label_fraction}" + ("_ssl" if args.pretrained else "_scratch")
    rows, test_probs = [], {}   # test_probs[model] = (y, prob) from the first seed

    for model_name in models_to_run:
        # Architecture cost is measured once per model, not once per seed.
        m0 = build_model(model_name).to(device)
        params = count_params(m0); flops = count_flops_g(m0, device)
        del m0; torch.cuda.empty_cache()

        per_seed = []
        for seed in args.seeds:
            set_seed(seed, args.deterministic)
            tr_ld, va_ld, te_ld = get_loaders(args.task, seed, args.label_fraction,
                                              deterministic=args.deterministic)
            model = build_model(model_name).to(device)
            if args.pretrained:
                load_pretrained_encoder(model, args.pretrained, device)
            print(f"\n[{model_name} | seed {seed}] params={params:.2f}M flops={flops}")
            model, best_val = train_model(model, tr_ld, va_ld, device,
                                          epochs=args.epochs)

            vprob, vy = predict(model, va_ld, device)
            thr = youden_threshold(vy, vprob)             # fixed on validation
            tprob, ty = predict(model, te_ld, device)
            mt = compute_metrics(ty, tprob, thr=thr)
            mt.update(model=model_name, seed=seed, params=params, flops=flops,
                      val_auroc=best_val)
            rows.append(mt); per_seed.append(mt["auroc"])
            # Persist per-sample test probabilities so reliability diagrams and
            # post-hoc pairwise tests do not require re-running the sweep.
            np.savez(C.RESULTS_DIR / f"probs_{tag}_{model_name}_seed{seed}.npz",
                     y=np.asarray(ty), prob=np.asarray(tprob))
            if seed == args.seeds[0]:
                test_probs[model_name] = (ty, tprob)
            # One checkpoint per model and seed. The tag carries the label
            # fraction and the initialisation, so a label-efficiency or SSL run
            # cannot overwrite the full-data scratch weights the Grad-CAM
            # figures are built from, and keeping every seed allows the
            # localisation measures to be repeated across seeds.
            torch.save({"model_state_dict": model.state_dict()},
                       C.CKPT_DIR / f"{model_name}_{tag}_seed{seed}.pth")
            print(f"   TEST auroc={mt['auroc']:.4f} acc={mt['acc']:.4f} "
                  f"sens={mt['sens']:.3f} spec={mt['spec']:.3f} f1={mt['f1']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS_DIR / f"ablation_{tag}_raw.csv", index=False)

    # ---- aggregate ----
    agg = (res.groupby("model")
              .agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
                   acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                   sens_mean=("sens", "mean"), spec_mean=("spec", "mean"),
                   f1_mean=("f1", "mean"),
                   ece_mean=("ece", "mean"), ece_std=("ece", "std"),
                   params=("params", "first"),
                   flops=("flops", "first"))
              .reindex(models_to_run))

    # Significance against DS-LiteDenseNet. Guarded so that a single-seed run,
    # or a --models subset omitting the reference, yields '--' rather than an
    # exception: the paired test needs more than one seed and DeLong needs the
    # reference probabilities.
    ref = "DSLiteDenseNet"
    sig = {m: ("--", "--") for m in models_to_run}
    ni  = {m: "--" for m in models_to_run}
    # Same verdict in the format Table 3 of the manuscript prints: the gap and
    # the paired t-test, not the non-inferiority test's own p.
    ni_tex = {m: "--" for m in models_to_run}
    ref_seed_auc = res[res.model == ref].sort_values("seed")["auroc"].to_numpy()
    if len(ref_seed_auc) > 0 and ref in test_probs:
        for mname in models_to_run:
            if mname == ref:
                continue
            a = res[res.model == mname].sort_values("seed")["auroc"].to_numpy()
            if len(a) > 1 and len(a) == len(ref_seed_auc):
                _, p_t = paired_ttest(ref_seed_auc, a)
                p_t = f"{p_t:.4f}"
                # Test whether DS-LiteDenseNet sits at most `ni_margin` AUROC
                # below this baseline.
                md, _, p_ni, is_ni = noninferiority_test(a, ref_seed_auc,
                                                         margin=args.ni_margin)
                ni[mname] = f"{'yes' if is_ni else 'no'} (d={md:+.3f},p={p_ni:.3f})"
                ni_tex[mname] = (f"{'yes' if is_ni else 'no'} "
                                 f"(${md:+.3f}$; {float(p_t):.3f})")
            else:
                p_t = "--"
            yb, pb = test_probs.get(mname, (None, None))
            if yb is not None:
                ya, pa = test_probs[ref]
                _, _, p_d = delong_roc_test(ya, pa, pb)
                p_d = f"{p_d:.4f}"
            else:
                p_d = "--"
            sig[mname] = (p_t, p_d)
    agg["p_paired_vs_ours"] = [sig.get(m, ("--", "--"))[0] for m in agg.index]
    agg["p_delong_vs_ours"] = [sig.get(m, ("--", "--"))[1] for m in agg.index]
    # Whether DS-LiteDenseNet is non-inferior to the model in this row, at the
    # pre-specified margin. Blank on its own row.
    agg[f"ours_noninf@{args.ni_margin}"] = [ni.get(m, "--") for m in agg.index]
    agg.to_csv(C.RESULTS_DIR / f"ablation_{tag}_summary.csv")
    print("\n", agg)

    # ---- LaTeX ----
    with open(C.RESULTS_DIR / f"ablation_{tag}.tex", "w") as f:
        f.write("\\begin{table}[ht]\\centering\\small\n")
        f.write(f"\\caption{{Ablation on {args.task} (locked test set, "
                f"mean$\\pm$std over {len(args.seeds)} seeds; threshold from val).}}\n")
        # Columns match Table 3 of the manuscript, ECE and the non-inferiority
        # verdict included.
        f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} l c c c c c c c c@{}}\n\\toprule\n")
        f.write("Model & AUROC & Acc & Sens & Spec & $F_1$ & ECE & Params(M) "
                "& NI? ($\\Delta$; $p_t$)\\\\\n\\midrule\n")
        for m in agg.index:
            r = agg.loc[m]
            f.write(f"{m} & {r.auroc_mean:.3f}$\\pm${r.auroc_std:.3f} "
                    f"& {r.acc_mean:.3f} & {r.sens_mean:.3f} & {r.spec_mean:.3f} "
                    f"& {r.f1_mean:.3f} & {r.ece_mean:.3f} & {r.params:.2f} "
                    f"& {ni_tex.get(m, '--')}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    print(f"\nWrote results to {C.RESULTS_DIR} (tag={tag})")


if __name__ == "__main__":
    main()
