# This plot-generating code is provided by generative AI tool. #

# Overlay several GP run logs: best training MSE (log scale) and, when the log has them, unique fraction and mean tree size.
# Usage (from the project root):
#   python3 run_dataset_3/compare_logs.py \
#       "first run=run_dataset_3/records/pop500_gen200_seed42.log" \
#       "after changes=run_dataset_3/records/boost2_pop500_gen200_seed42.log"
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

BASELINE = 636000   # constant-predictor MSE on dataset3 (training), used for the "% of baseline" labels
SHOW_BASELINE = False   # ✨ True = also draw the dashed grey constant-predictor line (it can hide curves that start near it)

GEN = re.compile(r"Generation:\s*(\d+)\s+Best training MSE:\s*([0-9.eE+\-]+|inf)")
UNI = re.compile(r"unique:\s*([0-9.]+)")
SIZ = re.compile(r"mean size:\s*([0-9.]+)")

def parse(path):
    gens, mse, uniq, size = [], [], [], []
    with open(path, errors="ignore") as f:
        for line in f:
            m = GEN.search(line)
            if not m:
                continue
            gens.append(int(m.group(1)))
            mse.append(float(m.group(2)))
            u, s = UNI.search(line), SIZ.search(line)
            uniq.append(float(u.group(1)) if u else None)
            size.append(float(s.group(1)) if s else None)
    return gens, mse, uniq, size

runs = []
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    runs.append((label, parse(path)))

# ✨ old logs have no "unique" / "mean size" columns -> then draw ONE clean panel instead of three (two of them empty)
has_extra = any(any(u is not None for u in r[1][2]) for r in runs)
if has_extra:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax0 = axes[0]
else:
    fig, ax0 = plt.subplots(1, 1, figsize=(7.5, 4.6))
    axes = [ax0]

highest_final = max(r[1][1][-1] for r in runs)
for label, (g, mse, uniq, size) in runs:
    line, = ax0.plot(g, mse, label=label)
    last = max((i for i in range(1, len(g)) if mse[i] < mse[i - 1] - 1e-9), default=0)
    ax0.plot([g[last]], [mse[last]], "o", color=line.get_color(), markersize=5)      # ✨ dot = last generation with an improvement
    below = len(runs) > 1 and mse[-1] == highest_final      # ✨ the highest curve gets its label BELOW the line, the others above, so labels never collide
    ax0.annotate(f"{mse[-1]:,.0f} ({100 * mse[-1] / BASELINE:.1f}% of baseline)", (g[-1], mse[-1]),
                 textcoords="offset points", xytext=(-4, -15 if below else 7), ha="right", fontsize=8, color=line.get_color())
    print(f"{label}: {len(g)} generations, final best MSE {mse[-1]:.1f}, last improvement at generation {g[last]}")
    if has_extra:
        if any(u is not None for u in uniq):
            axes[1].plot([a for a, u in zip(g, uniq) if u is not None], [u for u in uniq if u is not None], label=label)
        if any(s is not None for s in size):
            axes[2].plot([a for a, s in zip(g, size) if s is not None], [s for s in size if s is not None], label=label)

if SHOW_BASELINE:                                               # ✏️ was always drawn
    ax0.axhline(BASELINE, color="grey", ls="--", lw=1)
    ax0.text(0.01, 0.97, "constant predictor", transform=ax0.transAxes, va="top", color="grey", fontsize=8)
ax0.set_yscale("log")
ax0.yaxis.set_major_locator(LogLocator(subs=[1, 2, 5]))                              # ✨ readable ticks: 1k, 2k, 5k, 10k ...
ax0.yaxis.set_minor_locator(LogLocator(subs=[]))
ax0.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax0.set_title("Best training MSE (log scale)  •  dot = last improvement")
if has_extra:
    axes[1].set_title("Unique expressions / population")
    axes[2].set_title("Mean tree size")
for ax in axes:
    ax.set_xlabel("generation")
    ax.legend(fontsize=8)
ax0.set_xlabel("generation   (baseline = constant predictor, MSE 636,000)")   # ✨ says what "% of baseline" means
plt.tight_layout()
out = "run_dataset_3/plots/compare_logs.png"
plt.savefig(out, dpi=150)
print("Saved:", out)