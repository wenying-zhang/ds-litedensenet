"""Two questions the saved probability files can answer without any retraining.

  A screening operating point. The manuscript fixes thresholds with Youden's J,
  which weights a missed case and a false alarm equally. Screening for a
  transmissible disease does not, and guidance for tuberculosis triage asks for
  high sensitivity rather than a balanced point. So: hold sensitivity at a
  target on the validation split, carry that threshold to the test set unchanged
  as the main experiments do, and report what specificity survives.

  Calibration under prior shift. Training uses a class-balanced sampler, so the
  network learns an approximately even prior, and the external cohorts sit near
  4% positive. Some of the external calibration error is therefore arithmetic
  rather than a failure of transfer. Shifting the logit by the log prior ratio
  removes exactly that component:

      logit_adj = logit_raw + log(pi_test / (1 - pi_test))
                            - log(pi_train / (1 - pi_train))

  Whatever calibration error survives the shift is the part transfer is
  responsible for. Reporting both separates the two.

Usage:
    python operating_point.py --probs "results/probs_pneumonia_lf1.0_scratch_*.npz"
    python operating_point.py --probs "results/extprobs_*vindr*.npz" --train_prior 0.5
    python operating_point.py --probs "results/extprobs_*.npz" --target_sens 0.90

Files written by run_ablation.py hold internal test predictions; those written
by eval_external.py hold external ones and also carry the threshold that was
used. Reads only; writes a CSV when --out is given.
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from stats import expected_calibration_error


def threshold_at_sensitivity(y, prob, target):
    """Lowest-specificity-cost threshold reaching at least `target` sensitivity."""
    fpr, tpr, thr = roc_curve(y, prob)
    ok = np.where(tpr >= target)[0]
    if len(ok) == 0:
        return float(np.min(prob))
    return float(thr[ok[0]])


def at_threshold(y, prob, thr):
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return (tp / (tp + fn) if tp + fn else np.nan,
            tn / (tn + fp) if tn + fp else np.nan)


def prior_shift(prob, train_prior, test_prior, eps=1e-6):
    p = np.clip(prob, eps, 1 - eps)
    # A single-class cohort gives test_prior 0 or 1, and the log-odds of that is
    # infinite. Clip the priors on the same footing as the probabilities: the
    # result is meaningless either way, but it is a finite number and a printed
    # warning rather than a screen of divide-by-zero errors.
    tp = float(np.clip(test_prior, eps, 1 - eps))
    rp = float(np.clip(train_prior, eps, 1 - eps))
    logit = np.log(p / (1 - p))
    adj = (logit
           + np.log(tp / (1 - tp))
           - np.log(rp / (1 - rp)))
    return 1.0 / (1.0 + np.exp(-adj))


def parse_name(path):
    """Recover model, seed, run tag and evaluation source from a filename.

    The tag matters. results/ accumulates probability files from every task and
    every label fraction, so grouping on model alone silently averages
    pneumonia, tuberculosis and COVID-19 together, and averages the label
    fractions on top of that. The resulting means are not a quantity.
    """
    stem = Path(path).stem
    # Anchor the seed to the end of the name, allowing only eval_external.py's
    # --out_suffix after it. The character immediately after that underscore
    # has to be a letter, so a leftover such as ..._seed0_1.npz from an aborted
    # or exploratory run is still rejected rather than folded into seed 0,
    # where it would corrupt the mean. Underscores are allowed after that first
    # letter, so a suffix of several words such as _lung_opacity is read.
    # plot_reliability.py names its seeds for the same reason.
    m = re.search(r"^(?:ext)?probs_(.+?)_([A-Za-z0-9]+)_seed(\d+)"
                  r"(_[A-Za-z][A-Za-z0-9_]*)?$", stem)
    if m:
        tag, model, seed = m.group(1), m.group(2), int(m.group(3))
        suffix = m.group(4) or ""
    else:
        return None
    src = "internal"
    for cand in ("chexpert", "vindr", "montgomery", "shenzhen", "rsna"):
        if cand in tag:
            src = cand
            tag = tag.replace("_" + cand, "")
            break
    # The suffix is what distinguishes two runs of the same checkpoints under
    # different label definitions, so it belongs in the tag: grouping on the
    # tag alone would average them, and the mean of two label definitions is
    # not a quantity.
    return model, seed, tag + suffix, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True,
                    help="glob over .npz files holding y and prob")
    ap.add_argument("--target_sens", type=float, default=0.90,
                    help="sensitivity to hold; 0.90 follows screening guidance")
    ap.add_argument("--train_prior", type=float, default=0.5,
                    help="positive rate the sampler presented during training")
    ap.add_argument("--test_prior", type=float, default=None,
                    help="positive rate at evaluation (default: measured per file)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="restrict to these seeds; leftovers are skipped by name")
    ap.add_argument("--tag", default=None,
                    help="restrict to one run tag, e.g. pneumonia_lf1.0_scratch")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(args.probs))
    if not files:
        raise SystemExit(f"No files matched {args.probs}")

    rows, skipped = [], []
    for f in files:
        if args.tag and args.tag not in Path(f).stem:
            continue
        d = np.load(f)
        y = np.asarray(d["y"]).astype(int)
        prob = np.asarray(d["prob"]).astype(float)
        if len(np.unique(y)) < 2:
            print(f"  [skip] {Path(f).name}: one class only")
            continue
        parsed = parse_name(f)
        if parsed is None:
            print(f"  ! ignoring {Path(f).name}: name does not end in _seed<N>")
            skipped.append(Path(f).name)
            continue
        model, seed, tag, src = parsed
        if args.seeds is not None and seed not in args.seeds:
            print(f"  ! ignoring {Path(f).name}: seed outside --seeds {args.seeds}")
            skipped.append(Path(f).name)
            continue
        pi = float(y.mean()) if args.test_prior is None else args.test_prior

        # Sensitivity-constrained point. With no held-out split inside the file
        # the threshold is chosen on the same predictions it is scored on, so
        # the specificity is optimistic; it is a ceiling, not an estimate.
        thr = threshold_at_sensitivity(y, prob, args.target_sens)
        sens, spec = at_threshold(y, prob, thr)

        ece_raw = expected_calibration_error(y, prob)
        adj = prior_shift(prob, args.train_prior, pi)
        ece_adj = expected_calibration_error(y, adj)

        rows.append(dict(file=Path(f).name, model=model, seed=seed, tag=tag, source=src,
                         n=len(y), prevalence=pi,
                         thr_at_sens=thr, sens=sens, spec=spec,
                         ece_raw=ece_raw, ece_prior_adjusted=ece_adj,
                         ece_removed_by_prior=ece_raw - ece_adj))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(
            "No usable files. Every path matched by --probs was skipped: "
            "check the glob, and read the '! ignoring' lines above for "
            "names that do not end in _seed<N> or seeds outside --seeds.")
    if skipped:
        print(f"\n{len(skipped)} file(s) skipped by name; they are not in any mean below.")
    if df.duplicated(["tag", "source", "model", "seed"]).any():
        raise SystemExit("Two files map to the same run, source, model and seed. "
                         "Pass --seeds to name the intended set; averaging them "
                         "would double-count a single run.")
    pd.set_option("display.width", 220)
    print(f"\nSensitivity held at {args.target_sens:.2f}; "
          f"training prior assumed {args.train_prior:.2f}\n")
    print(df.drop(columns=["file"]).round(4).to_string(index=False))

    if len(df) > 1:
        # Group by tag as well as source, so runs from different tasks and
        # different label fractions are never averaged together.
        g = (df.groupby(["tag", "source", "model"])[["spec", "ece_raw", "ece_prior_adjusted"]]
               .agg(["mean", "std"]).round(4))
        print("\nMean over seeds, within each run:\n" + g.to_string())
        if df.tag.nunique() > 1:
            print(f"\n{df.tag.nunique()} distinct runs matched. Read each tag "
                  f"separately; a mean across tags is not a quantity.")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")
    print("\nece_prior_adjusted is the calibration error remaining after "
          "correcting for the difference between the training and evaluation "
          "prevalence; the difference between the two columns is the component "
          "attributable to that prior shift alone.")


if __name__ == "__main__":
    main()
