import numpy as np
import scipy.linalg as sci_lin
import matplotlib.pyplot as plt
import random

#
# All this stuff is bras instead of kets,
# because I don't think about the outer product operator making at all here yet
# but I guess the exposition is in kets
#

def gf2(arr):
    return np.mod(arr, 2)

def kron_and(pair):
    return np.kron(pair[0], pair[1])

def kron_xor(pair):
    return gf2(
            np.kron(pair[0], np.ones_like(pair[1])) +
            np.kron(np.ones_like(pair[0]), pair[1]))

def kron_or(pair):
    xor_res = kron_xor(pair)
    and_res = kron_and(pair)
    return gf2(xor_res + and_res)

def apply_op(op, pair):
    ops_to_funcs = {
            "*": kron_and,
            "+": kron_xor,
            "v": kron_or,
            }
    return ops_to_funcs[op](pair)

def expand(vecs, ops):
    curr_pair = (vecs[0], vecs[1])
    vecs.pop(0)
    while vecs:
        vecs.pop(0)
        curr_op = ops.pop(0)
        curr_res = apply_op(curr_op, curr_pair)
        if vecs:
            curr_pair = (curr_res, vecs[0])
    return curr_res

def sierpinski(order):
    root = np.array([[1,0], [1,1]])
    curr_res = root.copy()
    for curr_order in range(order - 1):
        curr_res = np.kron(curr_res, root)
    return curr_res

def zeroed_hadamard(card):
    res = sci_lin.hadamard(card)
    res[res < 0] = 0
    return res

def rand_vec():
    return random.choice(
            [np.array([0,0]),
            np.array([1,0]),
            np.array([0,1]),
            np.array([1,1])]
            ).copy()


def rand_op():
    return random.choice(["*", "v", "+"])

def rand_proplin(length):
    assert length >= 2, "Prop lin length >= 2 for now"
    vecs = [rand_vec() for _ in range(length)]
    ops = [rand_op() for _ in range(length - 1)]
    return (vecs, ops)


def apply_hadamard(vecs):
    """
    Applies it our quick way
    Ignores normalization
    """
    return [vec.dot(sci_lin.hadamard(2)) for vec in vecs]


if __name__ == "__main__":
    vecs = [[0,1], [0,1], [0,1]]
    ops = ["*", "+"]
    print(expand(vecs, ops))
