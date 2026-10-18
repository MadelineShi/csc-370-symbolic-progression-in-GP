import random
from tree import tree, random_tree


def check_structure(node, max_depth):
    if max_depth < 0:
        raise ValueError("Tree exceeds the maximum depth")

    # Dataset 2 has three possible variable leaves instead of one.
    if node.op in ("x1", "x2", "x3"):
        if len(node.children) != 0:
            raise ValueError("Variables should have no children")
        return

    if node.op == "const":
        if len(node.children) != 0:
            raise ValueError("Constant should have no children")

        # Dataset 2 constants are real-valued rather than integers.
        if type(node.value) is not float:
            raise ValueError("Dataset2 constants should be floats")
        return

    if node.op not in ("+", "-", "*", "/"):
        raise ValueError("Unknown operator")

    if len(node.children) != 2:
        raise ValueError("A binary operator needs two children")

    for child in node.children:
        check_structure(child, max_depth - 1)


def test_evaluate():
    # Dataset 2 test expression: (x1 + 2.5) * (x2 / x3).
    candidate = tree(
        "*",
        children=(
            tree("+", children=(tree("x1"), tree("const", value=2.5))),
            tree("/", children=(tree("x2"), tree("x3"))),
        ),
    )

    examples = [
        ((3.0, 8.0, 2.0), 22.0),
        ((-2.5, 5.0, 1.0), 0.0),
        ((1.5, 9.0, 3.0), 12.0),
    ]

    for inputs, expected in examples:
        prediction = candidate.evaluate(inputs)

        if prediction != expected:
            raise ValueError("Tree evaluation is incorrect")

    if str(candidate) != "((x1 + 2.5) * (x2 / x3))":
        raise ValueError("Expression display is incorrect")

    print("PASS: recursive evaluation and expression display")


def test_random_tree():
    # A fixed seed makes failures reproducible.
    random.seed(42)
    for max_depth in range(4):
        for _ in range(100):
            candidate = random_tree(max_depth)
            check_structure(candidate, max_depth)
    print("PASS: structure and depth of 400 random trees")

    print("Example random expressions:")
    for _ in range(5):
        print(random_tree(3))


if __name__ == "__main__":
    test_evaluate()
    test_random_tree()

