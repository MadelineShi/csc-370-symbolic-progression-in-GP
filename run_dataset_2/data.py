import csv
import random

def load_dataset2(filename):
    dataset = []

    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)  # skip header

        for row in reader:
            # dataset 2 has 3 inputs
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

    # cant divide by the x3 values that got rounded to 0
    training_data = [row for row in training_data if row[0][2] != 0]
    test_data = [row for row in test_data if row[0][2] != 0]

    # scale these since they are all really different sizes
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
