import random
import math
import copy 
from tree import tree, random_tree

BINARY_OPS = ["+", "-", "*", "/", "**"]

UNARY_OPS = [
    "sin", "cos", "exp", "log",
    "asin", "acos", "atan",
    "sqrt", "cbrt"
]

def check_structure(node):
    if node.op in ["x", "const"]:
        return node.left is None and node.right is None

    # Unary operators must have only a left child
    if node.op in UNARY_OPS:
        if node.left is None or node.right is not None:
            return False

        return check_structure(node.left)

    # Binary operators must have both children
    if node.op in BINARY_OPS:
        if node.left is None or node.right is None:
            return False

        return (
            check_structure(node.left)
            and check_structure(node.right)
        )

    return False


if __name__ == "__main__":
    checks = []

    # A known binary expression: x + 2
    binary = tree(
        "+",
        left=tree("x"),
        right=tree("const", value=2)
    )

    checks.append(("Addition", binary.evaluate(3) == 5))
    checks.append(("Binary string", str(binary) == "(x + 2)"))
    checks.append(("Binary size", binary.size() == 3))
    checks.append(("Binary depth", binary.depth() == 1))

    # A nested unary expression: sin(x + 2)
    unary = tree("sin", left=binary)

    checks.append(("Nested evaluation", unary.evaluate(-2) == 0))
    checks.append(("Unary string", str(unary) == "sin((x + 2))"))
    checks.append(("Unary size", unary.size() == 4))
    checks.append(("Unary depth", unary.depth() == 2))

    # Fractional power: x ** (1 / 2)
    power = tree(
        "**",
        left=tree("x"),
        right=tree(
            "/",
            left=tree("const", value=1),
            right=tree("const", value=2)
        )
    )

    checks.append((
        "Fractional power",
        math.isclose(power.evaluate(4), 2)
    ))

    # Each row is: operator, input, expected output
    unary_cases = [
        ("sin", 0, 0),
        ("cos", 0, 1),
        ("exp", 0, 1),
        ("log", 1, 0),
        ("asin", 0, 0),
        ("acos", 1, 0),
        ("atan", 0, 0),
        ("cbrt", -8, -2),
        ("cbrt", 0, 0)
    ]

    for op, input_value, expected in unary_cases:
        candidate = tree(op, left=tree("x"))
        actual = candidate.evaluate(input_value)

        passed = math.isclose(
            actual, expected,
            rel_tol=1e-9,
            abs_tol=1e-9
        )

        checks.append((op + " at " + str(input_value), passed))

    # Invalid unary calculations should return nan
    invalid_cases = [
        ("log", 0),
        ("log", -1),
        ("asin", 2),
        ("acos", -2),
        ("exp", 1000)
    ]

    for op, input_value in invalid_cases:
        candidate = tree(op, left=tree("x"))
        result = candidate.evaluate(input_value)

        checks.append((
            "Invalid " + op + " at " + str(input_value),
            math.isnan(result)
        ))

    # Invalid binary calculations
    division = tree(
        "/",
        left=tree("const", value=1),
        right=tree("const", value=0)
    )

    checks.append((
        "Division by zero",
        math.isnan(division.evaluate(0))
    ))

    checks.append((
        "Negative base with fractional power",
        math.isnan(power.evaluate(-4))
    ))

    # A deep copy should not change the original tree
    copied = copy.deepcopy(binary)
    copied.right.value = 99

    checks.append(("Original preserved", binary.evaluate(3) == 5))
    checks.append(("Copy changed", copied.evaluate(3) == 102))

    # Check 30 randomly generated trees
    random.seed(42)

    for i in range(30):
        candidate = random_tree(3)

        passed = (
            check_structure(candidate)
            and candidate.depth() <= 3
        )

        checks.append(("Random tree " + str(i), passed))

        if i < 3:
            print("Example:", str(candidate))

    # Report results
    failed = 0

    for name, passed in checks:
        if passed:
            print("PASS:", name)
        else:
            print("FAIL:", name)
            failed += 1

    if failed == 0:
        print("All tree checks passed!")
    else:
        print("Failed checks:", failed)
        raise SystemExit(1)