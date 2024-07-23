import numpy as np


def fib_sphere(n=1000):
    """
    Algorithm for producing directions from a Fibonacci sphere

    Inputs
    ------
    n: int
    Number of directions

    Returns
    -------
    xyz: array
    Array of 3D cartesian directions
    """
    goldenRatio = (1 + 5**0.5) / 2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / goldenRatio
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.array([x, y, z]).T
