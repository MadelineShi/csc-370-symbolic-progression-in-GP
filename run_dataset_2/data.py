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

    # Dataset 2 has rounded x3 values of 0 with large finite outputs, which cannot be learned using division.
    training_data = [row for row in training_data if row[0][2] != 0]
    test_data = [row for row in test_data if row[0][2] != 0]

    # Dataset 2 inputs have very different sizes, so scale them to about 0 to 1.
    x1_scale = max(abs(inputs[0]) for inputs, y in training_data)
    x2_scale = max(abs(inputs[1]) for inputs, y in training_data)
    x3_scale = max(abs(inputs[2]) for inputs, y in training_data)

    scaled_training_data = []
    for (x1, x2, x3), y in training_data:
        scaled_x1 = x1 / x1_scale
        scaled_x2 = x2 / x2_scale
        scaled_x3 = x3 / x3_scale

        scaled_training_data.append(((scaled_x1, scaled_x2, scaled_x3), y))

    scaled_test_data = []
    for (x1, x2, x3), y in test_data:
        scaled_x1 = x1 / x1_scale
        scaled_x2 = x2 / x2_scale
        scaled_x3 = x3 / x3_scale

        scaled_test_data.append(((scaled_x1, scaled_x2, scaled_x3), y))

    return scaled_training_data, scaled_test_data



if __name__ == "__main__":
    dataset2 = load_dataset2("data/dataset2.csv")

    training_data, test_data = split_dataset(dataset2)

    print("Dataset2 total rows:", len(dataset2))
    print("Training data rows:", len(training_data))
    print("Test data rows:", len(test_data))
