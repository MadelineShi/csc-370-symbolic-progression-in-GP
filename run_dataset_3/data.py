import csv
import random

def load_dataset3(filename):
    dataset = []

    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)  # skip the first row (the header)

        for row in reader:
            x = float(row[0])
            y = float(row[1])
            dataset.append((x, y))

    return dataset
    


def split_dataset(dataset, seed = 42):
    shuffled_data = dataset.copy()

    rng = random.Random(seed)
    rng.shuffle(shuffled_data)

    split_position = int(len(shuffled_data) * 0.8)

    training_data = shuffled_data[:split_position]
    test_data = shuffled_data[split_position:]

    return training_data, test_data



if __name__ == "__main__":
    dataset3 = load_dataset3("data/dataset3.csv")

    training_data, test_data = split_dataset(dataset1)

    print("Dataset 3 total rows:", len(dataset3))
    print("Training data rows:", len(training_data))
    print("Test data rows:", len(test_data))