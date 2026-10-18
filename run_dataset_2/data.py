import csv
import random

def load_dataset2(filename):
    dataset = []

    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)  # skip the first row (the header)

        for row in reader:
            # Dataset 2 has three inputs instead of the one input in Dataset 1.
            x1 = float(row[0])
            x2 = float(row[1])
            x3 = float(row[2])
            y = float(row[3])
            dataset.append(((x1, x2, x3), y))

    return dataset
    


def split_dataset(dataset, seed=42):
    shuffled_data = dataset.copy()

    rng = random.Random(seed)
    rng.shuffle(shuffled_data)

    split_position = int(len(shuffled_data) * 0.8)

    training_data = shuffled_data[:split_position]
    test_data = shuffled_data[split_position:]

    return training_data, test_data



if __name__ == "__main__":
    dataset2 = load_dataset2("data/dataset2.csv")

    training_data, test_data = split_dataset(dataset2)

    print("Dataset2 total rows:", len(dataset2))
    print("Training data rows:", len(training_data))
    print("Test data rows:", len(test_data))
