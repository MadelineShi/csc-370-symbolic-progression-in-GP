# Plot many seeds of one (or several) settings:
#   left : best training MSE vs generation, one thin line per seed + bold median (log scale)
#   right: final TEST MSE of every seed (one dot per seed) + median bar
# It also prints how many seeds "converged" (see CONVERGED_TOL below).
# Usage (from the project root):
#   python3 run_dataset_3/plot_multiseed.py "14 ops, boost 2=run_dataset_3/records/multiseed/fullops_boost4_pop200_gen100"
#   python3 run_dataset_3/plot_multiseed.py "8 ops=<dir1>" "14 ops=<dir2>"       # several settings side by side
import glob
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE = 636000          # constant-predictor MSE on the training set
CONVERGED_TOL = 0.01       # "converged" = best MSE improved by < 1% over the last 20% of generations
GEN = re.compile(r"Generation:\s*(\d+)\s+Best training MSE:\s*([0-9.eE+\-]+|inf)")
TEST = re.compile(r"Final test MSE:\s*([0-9.eE+\-]+|inf)")
COLORS = ["#2a6fbb", "#d9822b", "#3a9d5d", "#8a4fbf"]


def read_run(path):
    gens, mse, test = [], [], None
    for line in open(path, errors="ignore"):
        m = GEN.search(line)
        if m:
            gens.append(int(m.group(1))); mse.append(float(m.group(2)))
        t = TEST.search(line)
        if t:
            test = float(t.group(1))
    return np.array(gens), np.array(mse), test


def converged(mse):
    n = len(mse)
    k = int(0.8 * n)
    return mse[-1] >= (1 - CONVERGED_TOL) * mse[k]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
for idx, arg in enumerate(sys.argv[1:]):
    label, folder = arg.split("=", 1)
    color = COLORS[idx % len(COLORS)]
    runs = []
    for p in sorted(glob.glob(folder + "/seed*.log")):
        g, m, t = read_run(p)
        if len(g):
            runs.append((p, g, m, t))
    if not runs:
        print("no logs found in", folder); continue

    n = min(len(r[1]) for r in runs)
    curves = np.array([r[2][:n] for r in runs])
    gens = runs[0][1][:n]
    for c in curves:
        ax1.plot(gens, c, color=color, alpha=0.25, lw=1)
    ax1.plot(gens, np.median(curves, axis=0), color=color, lw=2.5, label=f"{label} (median of {len(runs)})")

    tests = np.array([r[3] for r in runs if r[3] is not None])
    xs = idx + np.linspace(-0.12, 0.12, len(tests))
    ax2.scatter(xs, tests, color=color, s=28, alpha=0.8)
    ax2.hlines(np.median(tests), idx - 0.25, idx + 0.25, color=color, lw=2.5)
    ax2.text(idx, np.median(tests) * 1.25, f"median {np.median(tests):,.0f}", ha="center", fontsize=8)

    conv = sum(converged(c) for c in curves)
    print(f"{label}: {len(runs)} seeds | converged (<{CONVERGED_TOL:.0%} gain in last 20% of gens): {conv}/{len(runs)}"
          f" | final test MSE: min {tests.min():,.0f}, median {np.median(tests):,.0f}, max {tests.max():,.0f}"
          f" | seeds under 5% of baseline: {(tests < 0.05 * BASELINE).sum()}/{len(tests)}")

ax1.axhline(BASELINE, color="grey", ls="--", lw=1)
ax1.text(0.01, 0.97, "constant predictor", transform=ax1.transAxes, va="top", color="grey", fontsize=8)
ax1.set_yscale("log"); ax1.set_xlabel("generation"); ax1.set_ylabel("best training MSE (log)")
ax1.set_title("Convergence of every seed"); ax1.legend(fontsize=8)
ax2.set_yscale("log"); ax2.set_ylabel("final test MSE (log)"); ax2.set_title("Final result per seed")
ax2.set_xticks(range(len(sys.argv) - 1)); ax2.set_xticklabels([a.split("=", 1)[0] for a in sys.argv[1:]], fontsize=8)
ax2.axhline(BASELINE, color="grey", ls="--", lw=1)
plt.tight_layout()
out = "run_dataset_3/plots/multiseed.png"
plt.savefig(out, dpi=150)
print("Saved:", out)