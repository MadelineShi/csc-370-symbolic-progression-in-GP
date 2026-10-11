from data import load_dataset1, split_dataset
from gp import evolve, fitness


if __name__ == "__main__":
    dataset1 = load_dataset1("data/dataset1.csv")
    training_data, test_data = split_dataset(dataset1, seed=42)

    print("Training rows:", len(training_data), flush=True)
    print("Test rows:", len(test_data), flush=True)

    population, fitness_scores = evolve(
        training_data,
        population_size=100, #10
        generations=50, #2
        initial_depth=3,
        max_depth=5,
        mutation_rate=0.2,
        seed=42
    )

    best_index = 0

    for i in range(1, len(population)):
        if fitness_scores[i] < fitness_scores[best_index]:
            best_index = i

    best_tree = population[best_index]

    print("Final expression:", str(best_tree), flush=True)
    print("Training MSE:", fitness_scores[best_index], flush=True)
    print("Test MSE:", fitness(best_tree, test_data), flush=True)