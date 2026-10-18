import io, contextlib
import numpy as np
import tree as tree_module            # module (not the class) so we can set BOOST
from data import load_dataset3, split_dataset
from gp import evolve, fitness

BOOSTS = [1.0, 2.0]         # 1.0 as uniform baseline
SEEDS = list(range(1, 21))
POP, GENS, INIT_DEPTH, MAX_DEPTH = 200, 100, 4, 5

if __name__ == "__main__":
    data = load_dataset3("data/dataset3.csv")
    training_data, test_data = split_dataset(data, seed = 42)

    print("boost | seed | train MSE | test MSE | expression", flush = True)
    results = {}
    for boost in BOOSTS:
        tree_module.BOOST = boost
        for seed in SEEDS:
            with contextlib.redirect_stdout(io.StringIO()):     # hide per-generation printing
                pop, scores = evolve(training_data, population_size = POP, generations = GENS,
                                     initial_depth = INIT_DEPTH, max_depth = MAX_DEPTH,
                                     mutation_rate = 0.2, seed = seed)
            b = int(np.argmin(scores))
            test = fitness(pop[b], test_data)
            results.setdefault(boost, []).append((scores[b], test))
            print(f"{boost:5} | {seed:4} | {scores[b]:12.1f} | {test:12.1f} | {str(pop[b])[:70]}", flush = True)

    print("\nboost | mean train | median train | best train | mean test")
    for boost, r in results.items():
        tr = [a for a, _ in r]; te = [b for _, b in r]
        print(f"{boost:5} | {np.mean(tr):10.1f} | {np.median(tr):12.1f} | {min(tr):10.1f} | {np.mean(te):9.1f}")