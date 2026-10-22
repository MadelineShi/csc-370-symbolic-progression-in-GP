import random

class tree:
    def __init__(self, op, children = (), value = None):
        self.op = op
        self.children = children
        self.value = value


    def __str__(self): # print the formula
        # dataset 2 has 3 x values
        if self.op in ("x1", "x2", "x3"):
            return self.op

        if self.op == "const":
            return str(self.value)

        left = str(self.children[0])
        right = str(self.children[1])

        return f"({left} {self.op} {right})"


    def evaluate(self, inputs):
        # unpack the 3 inputs
        x1, x2, x3 = inputs

        if self.op == "x1":
            return x1

        if self.op == "x2":
            return x2

        if self.op == "x3":
            return x3

        if self.op == "const":
            return self.value

        # evaluate both sides first
        left = self.children[0].evaluate(inputs)
        right = self.children[1].evaluate(inputs)

        if self.op == "+":
                return left + right

        if self.op == "-":
                return left - right

        if self.op == "*":
                return left * right

        if self.op == "/":
            # dont let division by something tiny crash the run
            if abs(right) < 0.000001:
                return 1.0
            return left / right

        raise ValueError("Unknown operator")

    
    def get_paths(self, path=()): # paths used for mutation and crossover
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
        if len(path) == 0: # this is the node to replace
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


    def depth(self):
        if len(self.children) == 0:
            return 0

        deepest_child = 0

        for child in self.children:
            child_depth = child.depth()

            if child_depth > deepest_child:
                deepest_child = child_depth

        return 1 + deepest_child


    def size(self):
        total = 1

        for child in self.children:
            total += child.size()

        return total


def random_tree(max_depth):
    if max_depth < 0:
        raise ValueError("max_depth cannot be negative")

    if max_depth == 0 or random.random() < 0.3:  # sometimes stop early and make a leaf
        # pick a variable or constant
        terminal = random.choice(["x1", "x2", "x3", "const"])

        if terminal in ("x1", "x2", "x3"):
            return tree(terminal, children = ())

        # dataset 2 can use decimal constants
        # wider range helps after scaling the inputs
        constant = random.uniform(-100, 100)
        return tree("const", children=(), value = constant)
    
    op = random.choice(["+", "-", "*", "/"])

    left = random_tree(max_depth - 1)
    right = random_tree(max_depth - 1)

    return tree(op, children=(left, right))


