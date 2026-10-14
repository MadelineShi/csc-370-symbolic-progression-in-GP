import random

from gp import (
    generate_population,
    evaluate_population,
    next_generation,
)
from test_tree import check_structure

random.seed(42)

# example y = 2x + 1 for checking the code
training_data = [(-2, -3), (-1, -1), (0, 1), (1, 3), (2, 5)]

population = generate_population(6, 2)

before = []
for candidate in population:
    check_structure(candidate, 2)
    before.append(str(candidate))

scores = evaluate_population(population, training_data)

# use 100% mutation here to test the code
new_population = next_generation(
    population, scores, max_depth=3, mutation_rate=1.0
)

if len(new_population) != 6:
    raise ValueError("Population size changed")

for index in range(len(population)):
    if str(population[index]) != before[index]:
        raise ValueError("An original tree was modified")

for candidate in new_population:
    check_structure(candidate, 3)

print("Old scores:", scores)
print("New scores:", evaluate_population(new_population, training_data))
print("PASS: one generation, valid structures, original trees preserved")