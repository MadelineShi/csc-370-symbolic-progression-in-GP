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


