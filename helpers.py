# -*- coding: utf-8 -*-
# Simple helper functions that aren't very project specific
import numpy as np


def estimate_parameters(mu, var):
    """Estimates the parameters of a beta distribution given the
    characteristics of a normal distribution
    (Way easier to play around with the values)

    """

    alpha = ((1 - mu) / var - 1/mu) * mu**2
    beta = alpha * (1 / mu - 1)
    return alpha, beta


def fcoin(bias):
    """Flips a true/false-coin with @bias to true
    """

    return np.random.binomial(1, bias)


def print3(T):
    # Prints 3dim Array T like this:
    # for each z, T[i,j,z] is a matrix with i rows and j columns
    print("Array with dimensions", str(np.shape(T)))
    for k in range(0, np.shape(T)[2]):
        print(T[:, :, k])
