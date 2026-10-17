"""Save this file in run_dataset_3, beside tree.py and data.py.

Run from the project root:
    python3 run_dataset_3/plot_dataset3_fit.py

Reads the completed single-run log; never runs GP. Uses the existing tree
evaluator, so operator semantics remain the same as in the experiment.
"""

import ast
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_dataset3, split_dataset
from tree import tree


# Change these only if the log filename or original split seed differs.
LOG_FILENAME = "pop500_gen200_seed42.log"
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
            elif line.startswith("Training MSE:"):
                logged_training_mse = float(line.split(":", 1)[1].strip())
            elif line.startswith("Test MSE:"):
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
        destination = output_folder / (log_path.stem + "_fit." + extension)
        fig.savefig(destination, dpi=200, bbox_inches="tight")
        print("Saved:", destination)
    plt.close(fig)


if __name__ == "__main__":
    main()