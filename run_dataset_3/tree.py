import numpy as np
import random
import math

BINARY_OPS = ["+", "-", "*", "/"]
UNARY_OPS = ["sin", "cos", "exp", "log"]

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
            return np.full(x.shape, float(self.value))   # change: array of the constant

        # With analytic functions, every operator needs at least its left child 
        left_value = self.left.evaluate(x)

        # Unary operators: only evaluate the left child
        if self.op == "sin":
            return np.sin(left_value)

        if self.op == "cos":
            return np.cos(left_value)

        if self.op == "exp":
            return np.exp(np.clip(left_value, -30.0, 30.0)) # clip the input so it won't overflow

        if self.op == "log":
            mag = np.abs(left_value)
            return np.log(np.where(mag < 1e-9, 1.0, mag)) # 注意 #


        if self.op not in BINARY_OPS: # check the remaining operators
            raise ValueError("Unknown operator: " + str(self.op))
        
        right_value = self.right.evaluate(x)

        if self.op == "+":
            return left_value + right_value

        if self.op == "-":
            return left_value - right_value

        if self.op == "*":
            return left_value * right_value
 
        if self.op == "/":
            tiny = np.abs(right_value) < 1e-6 # 这里也改！
            result = np.where(tiny, 1.0, left_value / np.where(tiny, 1.0, right_value))

        return np.clip(result, -1e12, 1e12)
        

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


def random_tree(max_depth, method = "grow", force_root_op = False):
    if max_depth < 0:
        raise ValueError("max_depth cannot be negative")

    stop_early = (method == "grow" and random.random() < 0.3)
    if max_depth == 0 or (stop_early and not force_root_op): 
        terminal = random.choice(["x", "const"])

        if terminal == "x":
            return tree("x")

        return tree("const", value = random.randint(-10, 10))
    
    op = choose_operator(BINARY_OPS + UNARY_OPS)

     # Every operator needs a left child
    left_child = random_tree(max_depth - 1)

    # Unary operators need only one child
    if op in UNARY_OPS:
        return tree(op, left=left_child)
    
     # Binary operators also need a right child
    right_child = random_tree(max_depth - 1)

    return tree(op, left=left_child, right=right_child)


def ramped_half_and_half(population_size, min_depth = 2, max_depth = 6):
    depths = list(range(min_depth, max_depth + 1))
    population = []
 
    for i in range(population_size):
        depth = depths[i % len(depths)]
        method = "full" if (i // len(depths)) % 2 == 0 else "grow"
        population.append(random_tree(depth, method, force_root_op = True))
 
    return population



def choose_operator(operators):  # micro managing!
    weights = []

    for op in operators:
        if op == "sin" or op == "exp":
            weights.append(1.5)
        else:
            weights.append(1.0)

    selected = random.choices(
        operators,
        weights=weights,
        k=1
    )

    return selected[0]
