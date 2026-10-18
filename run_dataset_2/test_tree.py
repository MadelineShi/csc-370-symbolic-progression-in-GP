import random
from tree import tree, random_tree


def check_structure(node, max_depth):
    if max_depth < 0:
        raise ValueError("Tree exceeds the maximum depth")

    if node.op == "x":
        if len(node.children) != 0:
            raise ValueError("x should have no children")
        return

    if node.op == "const":
        if len(node.children) != 0:
            raise ValueError("Constant should have no children")

        # this is solely for dataset 1
        if type(node.value) is not int:
            raise ValueError("Dataset1 constants should be integers")
        return

    if node.op not in ("+", "-", "*", "/"):
        raise ValueError("Unknown operator")

    if len(node.children) != 2:
        raise ValueError("A binary operator needs two children")

    for child in node.children:
        check_structure(child, max_depth - 1)


def test_evaluate():
    # Hand-built expression (x + 2) * x only for testing.
    candidate = tree(
        "*",
        children=(
            tree("+", children=(tree("x"), tree("const", value=2))),
            tree("x"),
        ),
    )
    
    for x in (-3, 0, 2.5):
        prediction = candidate.evaluate(x)
        expected = (x + 2) * x

        if prediction != expected:
            raise ValueError("Tree evaluation is incorrect")

    if str(candidate) != "((x + 2) * x)":
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

