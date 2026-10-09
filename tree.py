import numpy as np
import random

class tree:
    def __init__(self, op, children = (), value = None):
        self.op = op
        self.children = children
        self.value = value


    def __str__(self): # this is for testing if we are generating a legal formula of x!
        if self.op == "x":
            return "x"

        if self.op == "const":
            return str(self.value)

        left = str(self.children[0])
        right = str(self.children[1])

        return f"({left} {self.op} {right})"


    def evaluate(self,x):
        if self.op == "x":
            return x

        if self.op == "const":
            return self.value

        #if it is an operator, evalute its children!
        left = self.children[0].evaluate(x)
        right = self.children[1].evaluate(x)

        if self.op == "+":
                return left + right

        if self.op == "-":
                return left - right

        if self.op == "*":
                return left * right

        if self.op == "/":
            if right == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return left / right

        raise ValueError("Unknown operator")

    
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
        if len(path) == 0:
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


def random_tree(max_depth):
    if max_depth == 0 or random.random() < 0.3:  # we set the probability of creating a leaf node to 0.3
        terminal = random.choice(["x", "const"])

        if terminal == "x":
            return tree("x", children = ())

        constant = random.randint(-10, 10)
        return tree("const", children=(), value = constant)
    
    op = random.choice(["+", "-", "*", "/"])

    left = random_tree(max_depth - 1)
    right = random_tree(max_depth - 1)

    return tree(op, children=(left, right))


