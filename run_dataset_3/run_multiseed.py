# Run the SAME setting with many random seeds, save one log per seed + a summary table.
# Usage (from the project root, NOT from inside run_dataset_3):
#   python3 run_dataset_3/run_multiseed.py                 # one run at a time
#   python3 run_dataset_3/run_multiseed.py --workers 4     # 4 seeds in parallel (uses several CPU cores)
#
# Logs go to  run_dataset_3/records/multiseed/<TAG>/seed<N>.log   (same line format as your normal logs)
# Summary:    run_dataset_3/records/multiseed/<TAG>/summary.csv
# Safe to stop and restart: seeds whose log already contains "Final test MSE" are skipped.
import argparse
import contextlib
import os
import time
from multiprocessing import Pool

from data import load_dataset3, split_dataset
from gp import evolve, fitness

# ---------- the setting you want to repeat (edit here) ----------
TAG         = "fullops_boost2_pop200_gen400"   # name of this setting; used for the folder name
POP         = 200
GENS        = 400
INIT_DEPTH  = 4
MAX_DEPTH   = 8
MUTATION    = 0.2
SEEDS       = list(range(1, 11))               # 10 independent runs: seeds 1..10
SPLIT_SEED  = 42                               # the train/test split stays FIXED for every run
# -----------------------------------------------------------------

OUT_DIR = os.path.join("run_dataset_3", "records", "multiseed", TAG)


def log_path(seed):
    return os.path.join(OUT_DIR, f"seed{seed}.log")


def already_done(seed):
    p = log_path(seed)
    if not os.path.exists(p):
        return False
    with open(p, errors="ignore") as f:
        return "Final test MSE" in f.read()


def run_one(seed):
    if already_done(seed):
        return (seed, None, None, 0.0, "skipped (already done)")
    data = load_dataset3("data/dataset3.csv")
    training_data, test_data = split_dataset(data, seed=SPLIT_SEED)

    start = time.time()
    with open(log_path(seed), "w") as f, contextlib.redirect_stdout(f):
        population, scores = evolve(
            training_data,
            population_size=POP,
            generations=GENS,
            initial_depth=INIT_DEPTH,
            max_depth=MAX_DEPTH,
            mutation_rate=MUTATION,
            seed=seed,
        )
        best = min(range(len(population)), key=lambda i: scores[i])
        train_mse = scores[best]
        test_mse = fitness(population[best], test_data)
        print("Final expression:", str(population[best]))
        print("Final training MSE:", train_mse)
        print("Final test MSE:", test_mse)
    return (seed, train_mse, test_mse, time.time() - start, "done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{TAG}: pop {POP}, gens {GENS}, max_depth {MAX_DEPTH}, seeds {SEEDS[0]}..{SEEDS[-1]}, workers {args.workers}", flush=True)

    rows = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(run_one, SEEDS):
                print(f"seed {r[0]}: {r[4]}  train {r[1]}  test {r[2]}  ({r[3]:.0f}s)", flush=True)
                rows.append(r)
    else:
        for s in SEEDS:
            r = run_one(s)
            print(f"seed {r[0]}: {r[4]}  train {r[1]}  test {r[2]}  ({r[3]:.0f}s)", flush=True)
            rows.append(r)

    # summary.csv is rebuilt from the logs, so skipped seeds are included too
    import re
    with open(os.path.join(OUT_DIR, "summary.csv"), "w") as f:
        f.write("seed,train_mse,test_mse\n")
        for s in SEEDS:
            txt = open(log_path(s), errors="ignore").read()
            tr = re.search(r"Final training MSE:\s*(\S+)", txt)
            te = re.search(r"Final test MSE:\s*(\S+)", txt)
            f.write(f"{s},{tr.group(1) if tr else ''},{te.group(1) if te else ''}\n")
    print("Saved:", os.path.join(OUT_DIR, "summary.csv"))