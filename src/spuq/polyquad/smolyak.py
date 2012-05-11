__author__ = 'ezander'


def generic_integrate(func, points, weights):
    sum = None
    for x, w in zip(points, weights):
        val = w * func(x)
        if sum is None:
            sum = val
        else:
            sum += val
    return val

def smolyak_rule(rvs, order):
    weights = [1]
    points = [0] * len(rvs)
    return points, weights

def smolyak_integrate(func, rvs, order):
    points, weights = smolyak_rule(rvs, order)
    return generic_integrate(func, points, weights)

def tensor_rule(rvs, order):
    weights = [1]
    points = [0] * len(rvs)
    return points, weights

def tensor_integrate(func, rvs, order):
    points, weights = tensor_rule(rvs, order)
    return generic_integrate(func, points, weights)