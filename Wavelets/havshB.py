import numpy as np


def havshB(q):
    # Split an array into 2 halves in horizontal direction
    if len(q.shape) == 1:
        q = q.reshape(1, -1)
    _, h = q.shape
    a = q[:, :h//2]
    b = q[:, h//2:]
    return a, b


