import math
import traning_data
import random
from tree import random_tree


def fitness(candidate, training_data):
    if(len(training_data) == 0):
        raise ValueError("Training data is empty")

    total_error = 0

    for x, y in training_data:
        try:
            prediction = candidate.evaluate(x)
            squared_error = (prediction - y) ** 2

        except (ZeroDivisionError, OverflowError):
            return float("inf")

        if not math.isfinite(squared_error):
            return float("inf")

        total_error += squared_error

    return total_error / len(training_data)


def generate_population(population_size, max_depth):
    population = []

    for _ in range(population_size):
        tree = random_tree(max_depth)
        population.append(tree)
            
    return population


def tournament_selection(population, fitness_scores, tournament_size=3):
    if len(population) != len(fitness_scores):
        raise ValueError("Each tree needs one fitness score")

    if tournament_size < 1 or tournament_size > len(population):
        raise ValueError("Invalid tournament size")

    indices = random.sample( # Randomly select the index of a participating tree
        range(len(population)),
        tournament_size
    )

    best_index = indices[0]

    for index in indices[1:]:
        if fitness_scores[index] < fitness_scores[best_index]:
            best_index = index

    return population[best_index]
    

def evaluate_population(population, training_data):
    fitness_scores = []

    for candidate in population:
        score = fitness(candidate, training_data)
        fitness_scores.append(score)

    return fitness_scores


def mutate(candidate, max_depth):
    path = random.choice(candidate.get_paths())

    available_depth = max_depth - len(path) 

    if available_depth < 0:
        raise ValueError("Selected node exceeds the depth limit")

    replacement = random_tree(available_depth)

    mutated_tree = candidate.replace_subtree(path, replacement)

    return mutated_tree


def crossover(candidate_a, candidate_b):
    path_a = random.choice(candidate_a.get_paths())
    path_b = random.choice(candidate_b.get_paths())

    subtree_a = candidate_a.get_subtree(path_a)
    subtree_b = candidate_b.get_subtree(path_b)

    child_a = candidate_a.replace_subtree(path_a, subtree_b)
    child_b = candidate_b.replace_subtree(path_b, subtree_a)

    return child_a, child_b


def next_generation(population, fitness_scores, max_depth, mutation_rate):
    best_index = 0

    for index in range(1, len(population)):
        if fitness_scores[index] < fitness_scores[best_index]:
            best_index = index

    new_population = [population[best_index]]

    while len(new_population) < len(population):
        candidate_a = tournament_selection(population, fitness_scores)
        candidate_b = tournament_selection(population, fitness_scores)

        child_a, child_b = crossover(candidate_a, candidate_b)

        for child, original in (
            (child_a, candidate_a),
            (child_b, candidate_b)
        ):
            if child.depth() > max_depth:
                child = original

            if random.random() < mutation_rate:
                child = mutate(child, max_depth)

            new_population.append(child)

            if len(new_population) == len(population):
                break

    return new_population