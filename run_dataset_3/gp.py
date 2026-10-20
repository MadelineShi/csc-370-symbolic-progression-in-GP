import math
import random
import copy
import numpy as np
from tree import random_tree, ramped_half_and_half


def fitness_arrays(candidate, X, Y):
    with np.errstate(all = "ignore"):
        prediction = candidate.evaluate(X)
        mse = float(np.mean((prediction - Y) ** 2))
    if not math.isfinite(mse):
        return float("inf")
    return mse


def fitness(candidate, training_data):
    if(len(training_data) == 0):
        raise ValueError("Training data is empty")

    X = np.array([x for x, y in training_data], dtype = float)
    Y = np.array([y for x, y in training_data], dtype = float)
    return fitness_arrays(candidate, X, Y)


def generate_population(population_size, max_depth):
    return ramped_half_and_half(population_size, 2, max_depth)


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
    

def evaluate_population(population, X, Y):
    fitness_scores = []

    for candidate in population:
        score = fitness_arrays(candidate, X, Y)
        fitness_scores.append(score)

    return fitness_scores

def collect_nodes(node, mother=None, side=None, depth=0):
    locations = [(node, mother, side, depth)]

    if node.left is not None:
        locations.extend(
            collect_nodes(node.left, node, "left", depth + 1)
        )

    if node.right is not None:
        locations.extend(
            collect_nodes(node.right, node, "right", depth + 1)
        )

    return locations


def mutate(candidate, max_depth): # I changed this since we are using pointers now
    # Work on a copy ----Dr. R
    child = copy.deepcopy(candidate)

    locations = collect_nodes(child)
    node, mother, side, depth = random.choice(locations)

    available_depth = max_depth - depth
    replacement = random_tree(available_depth)

    if mother is None:
        return replacement

    if side == "left":
        mother.left = replacement
    else:
        mother.right = replacement

    return child


def crossover(candidate_a, candidate_b):
    child_a = copy.deepcopy(candidate_a)
    child_b = copy.deepcopy(candidate_b)

    node_a, mother_a, side_a, depth_a = random.choice(
        collect_nodes(child_a)
    )

    node_b, mother_b, side_b, depth_b = random.choice(
        collect_nodes(child_b)
    )

    if mother_a is None:
        child_a = node_b
    elif side_a == "left":
        mother_a.left = node_b
    else:
        mother_a.right = node_b

    if mother_b is None:
        child_b = node_a
    elif side_b == "left":
        mother_b.left = node_a
    else:
        mother_b.right = node_a

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
            if child.depth() > max_depth: # damn sets a hard limit of how my data can fit, we need to go deeper
                child = original

            if random.random() < mutation_rate:
                child = mutate(child, max_depth)

            new_population.append(child)

            if len(new_population) == len(population):
                break

    return new_population


def evolve(
    training_data,
    population_size=100,
    generations=50,
    initial_depth=3,
    max_depth=5,
    mutation_rate=0.2,
    seed=42
):
    random.seed(seed)

    X = np.array([x for x, y in training_data], dtype = float)
    Y = np.array([y for x, y in training_data], dtype = float)
    population = generate_population(population_size, initial_depth)

    for generation in range(generations + 1):
        fitness_scores = evaluate_population(population, X, Y)

        best_index = 0

        for i in range(1, len(population)):
            if fitness_scores[i] < fitness_scores[best_index]:
                best_index = i

        unique = len(set(str(t) for t in population)) / len(population)
        mean_size = sum(t.size() for t in population) / len(population)
        inf_frac = sum(1 for s in fitness_scores if s == float("inf")) / len(population)

        print(
            "Generation:", generation,
            "Best training MSE:", fitness_scores[best_index],
            "Expression:", str(population[best_index]),
             "| unique:", round(unique, 2),            # 
            "mean size:", round(mean_size, 1),         # 
            "inf:", round(inf_frac, 2),                # 新加的仨
            flush=True
        )

        if generation < generations:
            population = next_generation(
                population,
                fitness_scores,
                max_depth,
                mutation_rate
            )

    return population, fitness_scores