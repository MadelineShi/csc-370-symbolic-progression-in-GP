import random

from gp import (
    generate_population,
    evaluate_population,
    next_generation,
)
from test_tree import check_structure

random.seed(42)

# small fake dataset where y is just all 3 added
training_data = [
    ((1.0, 2.0, 3.0), 6.0),
    ((2.0, 4.0, 1.0), 7.0),
    ((-1.0, 3.0, 2.0), 4.0),
    ((5.0, 1.0, -2.0), 4.0),
    ((0.0, 2.0, 1.0), 3.0),
]

population = generate_population(6, 2)

before = []
for candidate in population:
    check_structure(candidate, 2)
    before.append(str(candidate))

scores = evaluate_population(population, training_data)

# mutate everything for this test
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
