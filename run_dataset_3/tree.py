import numpy as np
import random
import math

BINARY_OPS = ["+", "-", "*", "/", "**"]

UNARY_OPS = [
    "sin", "cos", "exp", "log",
    "asin", "acos", "atan",
    "sqrt", "cbrt"
]

class tree:
    def __init__(self, op, left = None, right = None, value = None):
        self.op = op
        self.left = left 
        self.right = right # more intuitive build-up thanks to Dr. R's suggestion
        self.value = value


    def __str__(self): # this is for testing if we are generating a legal formula of x!
        if self.op == "x":
            return "x"

        if self.op == "const":
            return str(self.value)

        left_text = str(self.left)
        
        if self.op in UNARY_OPS:
            return f"{self.op}({left_text})"
        
        if self.op in BINARY_OPS:
            right_text = str(self.right)
            return f"({left_text} {self.op} {right_text})"

        raise ValueError("Unknown operator: " + str(self.op))


    def evaluate(self,x):
        if self.op == "x":
            return x

        if self.op == "const":
            return self.value

        # With analytic functions, every operator needs at least its left child 
        left_value = self.left.evaluate(x)

        if not math.isfinite(left_value):
            return float("nan")

        # Unary operators: only evaluate the left child
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

        if self.op not in BINARY_OPS: # check the remaining operators
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


    
    def get_paths(self, path=()): # for mutation and crossover etc.
        paths = [path]

        for i in range(len(self.children)):
            child = self.children[i]
            child_paths = child.get_paths(path + (i,))
            paths.extend(child_paths)

        return paths


    def get_subtree(self, path):
        node = self

        for index in path:
            node = node.children[index]

        return node


    def replace_subtree(self, path, replacement):
        if len(path) == 0: # already arrive the node to be replaced
            return replacement

        index = path[0]
        remaining_path = path[1:]

        new_children = list(self.children)

        new_children[index] = self.children[index].replace_subtree(
            remaining_path, replacement
        )

        return tree(
            self.op,
            children=tuple(new_children),
            value=self.value
        )


    def size(self):
        total = 1

        if self.left is not None:
            total += self.left.size()

        if self.right is not None:
            total += self.right.size()

        return total

    def depth(self):
        if self.left is None and self.right is None:
            return 0

        left_depth = 0
        right_depth = 0

        if self.left is not None:
            left_depth = self.left.depth()

        if self.right is not None:
            right_depth = self.right.depth()

        return 1 + max(left_depth, right_depth)


def random_tree(max_depth):
    if max_depth < 0:
        raise ValueError("max_depth cannot be negative")

    if max_depth == 0 or random.random() < 0.3:  # we set the probability of creating a leaf node to 0.3
        terminal = random.choice(["x", "const"])

        if terminal == "x":
            return tree("x")

        return tree("const", value = random.randint(-10, 10))
    
    op = random.choice(BINARY_OPS + UNARY_OPS)

     # Every operator needs a left child
    left_child = random_tree(max_depth - 1)

    # Unary operators need only one child
    if op in UNARY_OPS:
        return tree(op, left=left_child)
    
     # Binary operators also need a right child
    right_child = random_tree(max_depth - 1)

    return tree(op, left=left_child, right=right_child)


