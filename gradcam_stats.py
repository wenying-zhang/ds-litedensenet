"""Recompute every statistic quoted in Section 4.4 from the Grad-CAM raw CSVs.

run_gradcam.py writes one row per scored image. This script consumes those rows
and reproduces the numbers in Table 6 and the significance tests in the text, so
that no value in that section has to be derived by hand.

Two families of test are reported.

  Between models, on pneumonia. The two models are scored on the same images, so
  the annotated positives pair one-to-one by image_id. Continuous measures use a
  paired t-test with a Wilcoxon signed-rank test alongside it, because the
  measures are bounded fractions and the two tests can disagree; the pointing
  hit is binary and uses an exact McNemar test on the discordant pairs.

  Between classes, within a model. Negatives and positives are different images,
  so this is an unpaired Welch t-test, with Mann-Whitney reported beside it. For
  a non-significant contrast the script also prints the 95% interval and the
  effect detectable at 80% power, since "no difference found" on a small sample
  is not the same claim as "no difference".

Usage:
    python gradcam_stats.py                       # pneumonia pair + TB
    python gradcam_stats.py --task tb
    python gradcam_stats.py --model-a DSLiteDenseNet --model-b DenseNet121

Reads results/gradcam_<model>_<task>_raw.csv; writes nothing.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config as C


def _load(model, task):
    path = Path(C.RESULTS_DIR) / f"gradcam_{model}_{task}_raw.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}\nRun run_gradcam.py --task {task} first.")
    return pd.read_csv(path)


def class_contrast(df, column, label="lung"):
    """Welch test between negative and positive cases for one model."""
    if column not in df.columns:
        print(f"  {label:<8} not measured in this run (column absent)")
        return
    neg = df.loc[df.label == 0, column].dropna().to_numpy(float)
    pos = df.loc[df.label == 1, column].dropna().to_numpy(float)
    if len(neg) < 2 or len(pos) < 2:
        print(f"  {label:<8} too few measured cases ({len(neg)}/{len(pos)})")
        return
    diff = pos.mean() - neg.mean()
    t, p = stats.ttest_ind(pos, neg, equal_var=False)
    u, p_mw = stats.mannwhitneyu(pos, neg, alternative="two-sided")

    se = np.sqrt(pos.var(ddof=1) / len(pos) + neg.var(ddof=1) / len(neg))
    dof = se ** 4 / ((pos.var(ddof=1) / len(pos)) ** 2 / (len(pos) - 1)
                     + (neg.var(ddof=1) / len(neg)) ** 2 / (len(neg) - 1))
    half = stats.t.ppf(0.975, dof) * se
    # Effect this sample would detect at 80% power, two-sided alpha 0.05.
    mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * se

    print(f"  {label:<8} (-)={neg.mean():.4f}  (+)={pos.mean():.4f}  "
          f"delta={diff:+.4f}")
    print(f"           Welch p={p:.3e}   Mann-Whitney p={p_mw:.3e}   "
          f"n={len(neg)}/{len(pos)}")
    if p >= 0.05:
        print(f"           95% CI [{diff - half:+.4f}, {diff + half:+.4f}];  "
              f"detectable at 80% power: {mde:.4f}")


def area_adjusted_contrast(df):
    """Lung-energy contrast measured as excess over the area actually available.

    lung_energy is compared against lung_area, because a map with no spatial
    information integrates to exactly the mask's area fraction. That reference
    is not a constant: if the segmented lung is larger on positives than on
    negatives, an uninformative map gains lung energy between the classes and a
    raw delta credits the model for it. Subtracting the per-image area first
    removes that, and on tuberculosis it changes the reading.
    """
    if "lung_area" not in df.columns or "lung_energy" not in df.columns:
        return
    d = df.dropna(subset=["lung_energy", "lung_area"])
    if d.empty:
        return
    exc = d.lung_energy - d.lung_area
    en, ep = exc[d.label == 0], exc[d.label == 1]
    an, ap = d.lung_area[d.label == 0].mean(), d.lung_area[d.label == 1].mean()
    if len(en) < 2 or len(ep) < 2:
        return
    t, p = stats.ttest_ind(ep, en, equal_var=False)
    print(f"  {'lung-area':<8} (-)={an:.4f}  (+)={ap:.4f}  delta={ap - an:+.4f}"
          f"   <- the null for the lung-energy delta above")
    print(f"  {'excess':<8} (-)={en.mean():+.4f}  (+)={ep.mean():+.4f}  "
          f"delta={ep.mean() - en.mean():+.4f}")
    print(f"           Welch p={p:.3e}   (lung energy in excess of mask area)")


def paired_between_models(a, b, name_a, name_b):
    """Paired tests over the annotated positives common to both models."""
    a = a[a.label == 1].sort_values("image_id").reset_index(drop=True)
    b = b[b.label == 1].sort_values("image_id").reset_index(drop=True)
    if not a.image_id.equals(b.image_id):
        raise SystemExit("The two files do not cover the same positive images.")
    print(f"\nPaired over {len(a)} annotated positives "
          f"({name_b} minus {name_a}):")

    # run_gradcam.py writes box_energy and pointing_hit only for annotated RSNA
    # positives, and lung_energy only when a mask was produced, so on any task
    # but pneumonia those columns are absent from the CSV entirely rather than
    # present and null. Test for the column before testing its contents.
    for col, label in [("lung_energy", "lung energy"),
                       ("box_energy", "box energy"),
                       ("bg_energy", "border energy")]:
        if col not in a.columns or col not in b.columns:
            continue
        # A column can also be present but null on some rows: box_energy and
        # pointing_hit are written only for positives that carry a box, so an
        # unannotated positive leaves them empty. scipy returns nan on such a
        # column rather than raising, which would print a silent nan p-value.
        ok = a[col].notna().to_numpy() & b[col].notna().to_numpy()
        if not ok.any():
            continue
        av, bv = a.loc[ok, col].to_numpy(float), b.loc[ok, col].to_numpy(float)
        d = bv - av
        _, p_t = stats.ttest_rel(bv, av)
        _, p_w = stats.wilcoxon(bv, av)
        note = "" if ok.all() else f"   [{int(ok.sum())}/{len(ok)} pairs]"
        print(f"  {label:<14} {d.mean():+.4f}   paired t p={p_t:.3e}   "
              f"Wilcoxon p={p_w:.3e}{note}")

    if "pointing_hit" in a.columns and "pointing_hit" in b.columns:
        # astype(int) raises on a single null, so pair on the rows where both
        # models recorded a hit rather than on any row that has one.
        ok = a.pointing_hit.notna().to_numpy() & b.pointing_hit.notna().to_numpy()
    else:
        ok = None
    if ok is not None and ok.any():
        ha = a.loc[ok, "pointing_hit"].astype(int).to_numpy()
        hb = b.loc[ok, "pointing_hit"].astype(int).to_numpy()
        only_a = int(((ha == 1) & (hb == 0)).sum())
        only_b = int(((ha == 0) & (hb == 1)).sum())
        n_disc = only_a + only_b
        if n_disc == 0:
            # McNemar is a test on the discordant pairs alone. With none, the
            # two models agree on every image and there is nothing to test;
            # binomtest raises on n=0 rather than returning 1.
            print(f"  {'pointing':<14} {ha.mean() - hb.mean():+.4f}   "
                  f"discordant 0 vs 0   McNemar exact p=n/a "
                  f"(the two models agree on all {len(ha)} images)")
        else:
            p = stats.binomtest(only_a, n_disc, 0.5).pvalue
            print(f"  {'pointing':<14} {ha.mean() - hb.mean():+.4f}   "
                  f"discordant {only_a} vs {only_b}   McNemar exact p={p:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="both", choices=["pneumonia", "tb", "both"])
    ap.add_argument("--model-a", default="DSLiteDenseNet",
                    help="the compact model; paired tests report B minus A")
    ap.add_argument("--model-b", default="DenseNet121",
                    help="the baseline; omit the pairing by passing --model-b ''")
    args = ap.parse_args()

    tasks = ["pneumonia", "tb"] if args.task == "both" else [args.task]
    for task in tasks:
        print(f"\n=== {task} ===")
        a = _load(args.model_a, task)
        print(f"{args.model_a}:")
        class_contrast(a, "lung_energy", "lung")
        area_adjusted_contrast(a)
        class_contrast(a, "bg_energy", "border")

        if not args.model_b:
            continue
        try:
            b = _load(args.model_b, task)
        except SystemExit:
            print(f"({args.model_b} not scored on {task}; "
                  f"skipping the paired comparison)")
            continue
        print(f"{args.model_b}:")
        class_contrast(b, "lung_energy", "lung")
        area_adjusted_contrast(b)
        class_contrast(b, "bg_energy", "border")
        paired_between_models(a, b, args.model_a, args.model_b)


if __name__ == "__main__":
    main()
