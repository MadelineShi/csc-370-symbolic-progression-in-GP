from data import load_dataset3, split_dataset

dataset3 = load_dataset3("data/dataset3.csv")
training_data, test_data = split_dataset(dataset3, seed=42)

total_y = 0
for x, y in training_data:
    total_y += y

mean_y = total_y / len(training_data)

total_error = 0
for x, y in training_data:
    total_error += (mean_y - y) ** 2

baseline_mse = total_error / len(training_data)

print("Mean y:", mean_y)
print("Baseline training MSE:", baseline_mse)
print("Our GP training MSE:", 532210.5905218206)