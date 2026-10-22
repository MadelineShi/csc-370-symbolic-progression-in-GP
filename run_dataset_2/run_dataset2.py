from data import load_dataset2, split_dataset
from gp import evolve, mean_squared_error, normalized_mean_squared_error


if __name__ == "__main__":
    # load the 3 inputs and y
    dataset2 = load_dataset2("data/dataset2.csv")
    training_data, test_data = split_dataset(dataset2, seed=42)

    print("Training rows:", len(training_data), flush=True)
    print("Test rows:", len(test_data), flush=True)

    population, fitness_scores = evolve(
        training_data,
        population_size = 100,
        generations = 50,
        initial_depth = 3,
        max_depth = 5,
        mutation_rate = 0.2,
        seed = 44
    )

    best_index = 0

    for i in range(1, len(population)):
        if fitness_scores[i] < fitness_scores[best_index]:
            best_index = i

    best_tree = population[best_index]

    print("Final expression:", str(best_tree), flush=True)
    print("Training MSE:", mean_squared_error(best_tree, training_data), flush=True)
    print("Test MSE:", mean_squared_error(best_tree, test_data), flush=True)
    print("Normalized training MSE:", normalized_mean_squared_error(best_tree, training_data), flush=True)
    print("Normalized test MSE:", normalized_mean_squared_error(best_tree, test_data), flush=True)
