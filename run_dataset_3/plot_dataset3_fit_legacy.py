# This plot-generating code is provided by generative AI tool. #

"""Replot the OLD scalar-evaluator baseline; does not run GP.

Save beside data.py in run_dataset_3, then run from the project root:
    python3 -u run_dataset_3/plot_dataset3_fit_legacy.py

Requires the original dataset3.csv and the original data split implementation.
The historical evaluator is embedded, leaving current tree.py untouched.
Checks recomputed training and test MSE against the log before saving.
"""

import ast
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_dataset3, split_dataset
# Historical scalar evaluator: matches the pre-NumPy implementation.
# Kept here so this plotting script never imports or changes today's tree.py.
BINARY_OPS = ["+", "-", "*", "/", "**"]
UNARY_OPS = ["sin", "cos", "exp", "log", "asin", "acos", "atan", "sqrt", "cbrt"]


class tree:
    def __init__(self, op, left=None, right=None, value=None):
        self.op = op
        self.left = left
        self.right = right
        self.value = value

    def evaluate(self, x):
        if self.op == "x":
            return x
        if self.op == "const":
            return self.value

        left_value = self.left.evaluate(x)
        if not math.isfinite(left_value):
            return float("nan")

        if self.op == "sin":
            return math.sin(left_value)
        if self.op == "cos":
            return math.cos(left_value)
        if self.op == "exp":
            try:
                return math.exp(left_value)
            except OverflowError:
                return float("nan")
        if self.op == "log":
            if left_value <= 0:
                return float("nan")
            return math.log(left_value)
        if self.op == "asin":
            if left_value < -1 or left_value > 1:
                return float("nan")
            return math.asin(left_value)
        if self.op == "acos":
            if left_value < -1 or left_value > 1:
                return float("nan")
            return math.acos(left_value)
        if self.op == "atan":
            return math.atan(left_value)
        if self.op == "sqrt":
            if left_value < 0:
                return float("nan")
            return math.sqrt(left_value)
        if self.op == "cbrt":
            if left_value < 0:
                return -((-left_value) ** (1 / 3))
            return left_value ** (1 / 3)

        if self.op not in BINARY_OPS:
            raise ValueError("Unknown operator: " + str(self.op))
        right_value = self.right.evaluate(x)
        if not math.isfinite(right_value):
            return float("nan")
        if self.op == "+":
            return left_value + right_value
        if self.op == "-":
            return left_value - right_value
        if self.op == "*":
            return left_value * right_value
        if self.op == "/":
            if right_value == 0:
                return float("nan")
            return left_value / right_value
        if self.op == "**":
            try:
                return math.pow(left_value, right_value)
            except (ValueError, OverflowError):
                return float("nan")



# Change these only if the log filename or original split seed differs.
LOG_FILENAME = "records/pop500_gen200_seed42.log"
DATA_SPLIT_SEED = 42


def rebuild_tree(node):
    """Convert the printed expression into the existing tree class.

    This reads mathematical syntax only; it does not execute Python code.
    """
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ValueError("Only numeric constants are allowed")
        return tree("const", value=node.value)

    if isinstance(node, ast.Name) and node.id == "x":
        return tree("x")

    # Python parses a printed negative constant, e.g. -10, as unary minus.
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            value = node.operand.value
            if type(value) in (int, float):
                return tree("const", value=-value)
        raise ValueError("Unsupported unary expression")

    if isinstance(node, ast.BinOp):
        operator_names = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*",
            ast.Div: "/", ast.Pow: "**",
        }
        op = operator_names.get(type(node.op))
        if op is None:
            raise ValueError("Unsupported binary operator")
        left_child = rebuild_tree(node.left)
        right_child = rebuild_tree(node.right)
        return tree(op, left=left_child, right=right_child)

    if isinstance(node, ast.Call):
        allowed_functions = [
            "sin", "cos", "exp", "log", "asin", "acos", "atan", "sqrt", "cbrt"
        ]
        if (isinstance(node.func, ast.Name)
                and node.func.id in allowed_functions
                and len(node.args) == 1 and not node.keywords):
            return tree(node.func.id, left=rebuild_tree(node.args[0]))
        raise ValueError("Unsupported function call")

    raise ValueError("The log contains an unsupported expression")


def read_result(log_path):
    expression = None
    logged_training_mse = None
    logged_test_mse = None
    expression_count = 0
    with log_path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line.startswith("Final expression:"):
                expression = line.split(":", 1)[1].strip()
                expression_count += 1
            elif line.startswith(("Training MSE:", "Final training MSE:")):
                logged_training_mse = float(line.split(":", 1)[1].strip())
            elif line.startswith(("Test MSE:", "Final test MSE:")):
                logged_test_mse = float(line.split(":", 1)[1].strip())
    if expression_count != 1:
        raise ValueError("Use a log containing exactly one completed run")
    if logged_training_mse is None or logged_test_mse is None:
        raise ValueError("The completed run's training/test MSE footer is missing")
    return expression, logged_training_mse, logged_test_mse


def main():
    folder = Path(__file__).resolve().parent
    project_root = folder.parent
    log_path = folder / LOG_FILENAME
    if not log_path.is_file():
        raise FileNotFoundError(f"Log not found: {log_path}. Check LOG_FILENAME.")

    expression, logged_training_mse, logged_test_mse = read_result(log_path)
    candidate = rebuild_tree(ast.parse(expression, mode="eval").body)

    print("Reading data and evaluating the saved expression (no GP training)...", flush=True)
    dataset3 = load_dataset3(str(project_root / "data" / "dataset3.csv"))
    training_data, test_data = split_dataset(dataset3, seed=DATA_SPLIT_SEED)
    if not training_data:
        raise ValueError("Training data is empty")

    # Sort complete (x, y) pairs so the prediction line moves from left to right.
    ordered_data = sorted(training_data)
    x_values = []
    actual_values = []
    predictions = []
    residuals = []
    total_error = 0.0
    for x, y in ordered_data:
        prediction = candidate.evaluate(x)
        if not math.isfinite(prediction):
            raise ValueError(f"Non-finite prediction at training x={x}")
        residual = prediction - y
        x_values.append(x)
        actual_values.append(y)
        predictions.append(prediction)
        residuals.append(residual)
        total_error += residual ** 2

    recomputed_mse = total_error / len(training_data)
    print("Expression:", expression)
    print("Training rows:", len(training_data))
    print("Test rows:", len(test_data))
    print("Logged training MSE:", logged_training_mse)
    print("Recomputed training MSE:", recomputed_mse)
    print("Logged test MSE:", logged_test_mse)
    if not math.isclose(recomputed_mse, logged_training_mse, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(
            "MSE does not match the log. Check the CSV, split function/seed, "
            "and tree evaluator before comparing plots."
        )

    if not test_data:
        raise ValueError("Test data is empty")
    test_error = 0.0
    for x, y in test_data:
        prediction = candidate.evaluate(x)
        if not math.isfinite(prediction):
            raise ValueError(f"Non-finite prediction at test x={x}")
        test_error += (prediction - y) ** 2
    recomputed_test_mse = test_error / len(test_data)
    print("Recomputed test MSE:", recomputed_test_mse, flush=True)
    if not math.isclose(recomputed_test_mse, logged_test_mse, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(
            "Test MSE does not match the log. Check the original data split "
            "and evaluator; do not change the old expression to force a match."
        )

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    axes[0].scatter(x_values, actual_values, s=5, alpha=0.35,
                    color="#3274B5", label="Observed training data", rasterized=True)
    axes[0].plot(x_values, predictions, color="#D54436", linewidth=1.8,
                 label="GP champion prediction")
    axes[0].set_title("Dataset 3: observed data and GP champion")
    axes[0].set_ylabel("y")
    axes[0].legend()

    # The overview's large y range can hide differences on the right-hand side.
    right_x = []
    right_y = []
    right_predictions = []
    for index in range(len(x_values)):
        if x_values[index] >= 0:
            right_x.append(x_values[index])
            right_y.append(actual_values[index])
            right_predictions.append(predictions[index])
    if right_x:
        axes[1].scatter(right_x, right_y, s=5, alpha=0.35,
                        color="#3274B5", rasterized=True)
        axes[1].plot(right_x, right_predictions, color="#D54436", linewidth=1.8)
        axes[1].set_title("Right-hand detail: training inputs x >= 0 (separate y scale)")
    else:
        axes[1].text(0.5, 0.5, "No training inputs x >= 0", ha="center",
                      transform=axes[1].transAxes)
    axes[1].set_ylabel("y")

    axes[2].scatter(x_values, residuals, s=5, alpha=0.35,
                    color="#7147A8", rasterized=True)
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Training residuals: prediction minus observation")
    axes[2].set_ylabel("Residual")
    for ax in axes:
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"{log_path.stem} | training MSE = {recomputed_mse:,.2f}", fontsize=11
    )
    output_folder = folder / "plots"
    output_folder.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        destination = output_folder / (log_path.stem + "_legacy_fit." + extension)
        fig.savefig(destination, dpi=400, bbox_inches="tight")
        print("Saved:", destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
