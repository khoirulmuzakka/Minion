import numpy as np

def sphere(X):
    result = np.sum(X**2, axis=1)
    return result

def rosenbrock(X):
    result = np.sum(100.0 * (X[:, 1:] - X[:, :-1]**2.0)**2.0 + (1 - X[:, :-1])**2.0, axis=1)
    return result

def rastrigin(X):
    result = np.sum(X**2 - 10.0 * np.cos(2.0 * np.pi * X) + 10, axis=1)
    return result

def griewank(X):
    result = np.sum(X**2, axis=1) / 4000.0 - np.prod(np.cos(X / np.sqrt(np.arange(1, X.shape[1] + 1))), axis=1) + 1
    return result

def ackley(X):
    result = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(X**2, axis=1) / X.shape[1])) - \
             np.exp(np.sum(np.cos(2.0 * np.pi * X), axis=1) / X.shape[1]) + 20 + np.e
    return result

def zakharov(X):
    result = np.sum(X**2, axis=1) + np.sum(0.5 * np.arange(1, X.shape[1] + 1) * X, axis=1)**2 + \
             np.sum(0.5 * np.arange(1, X.shape[1] + 1) * X, axis=1)**4
    return result

def michalewicz(X):
    result = -np.sum(np.sin(X) * (np.sin(np.arange(1, X.shape[1] + 1) * X**2 / np.pi))**(2 * 10), axis=1)
    return result

def levy(X):
    w = 1 + (X - 1) / 4
    term1 = (np.sin(np.pi * w[:, 0]))**2
    term2 = np.sum((w[:, :-1] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[:, :-1] + 1))**2), axis=1)
    term3 = (w[:, -1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[:, -1]))**2)
    result = term1 + term2 + term3
    return result

def dixon_price(X):
    result = (X[:, 0] - 1)**2 + np.sum(np.arange(2, X.shape[1] + 1) * (2 * X[:, 1:]**2 - X[:, :-1])**2, axis=1)
    return result

# Additional test functions

def bent_cigar(X):
    result = X[:, 0]**2 + 1e6 * np.sum(X[:, 1:]**2, axis=1)
    return result

def discus(X):
    result = 1e6 * X[:, 0]**2 + np.sum(X[:, 1:]**2, axis=1)
    return result

def weierstrass(X):
    a = 0.5
    b = 3
    k_max = 20
    n = X.shape[1]
    result = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        inner_sum = np.zeros(X.shape[1])
        for k in range(k_max):
            inner_sum += a**k * np.cos(2 * np.pi * b**k * (X[i] + 0.5))
        result[i] = np.sum(inner_sum) - n * np.sum([a**k * np.cos(np.pi * b**k) for k in range(k_max)])
    return result

def happy_cat(X):
    result = ((np.sum(X**2, axis=1) - X.shape[1])**2)**0.25 + (0.5 * np.sum(X**2, axis=1) + np.sum(X, axis=1)) / X.shape[1] + 0.5
    return result

def hgbat(X):
    result = ((np.sum(X**2, axis=1)**2)**0.25 + (0.5 * np.sum(X**2, axis=1) + np.sum(X, axis=1)) / X.shape[1] + 0.5)
    return result

def hcf(X):
    result = np.sum(np.abs(X), axis=1) * np.exp(np.sum(np.abs(X), axis=1) / X.shape[1])
    return result

def grie_rosen(X):
    result = 1 + np.sum(100 * (X[:, 1:] - X[:, :-1]**2)**2 + (1 - X[:, :-1])**2, axis=1)
    return result

def scaffer6(X):
    result = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        for j in range(X.shape[1] - 1):
            term1 = X[i, j]**2 + X[i, j+1]**2
            sin_term = np.sin(np.sqrt(term1))**2
            denom_term = (1 + 0.001 * term1)**2
            result[i] += 0.5 + (sin_term - 0.5) / denom_term
    return result

def hybrid_composition1(X):
    result = sphere(X) + rosenbrock(X) + rastrigin(X)
    return result

def hybrid_composition2(X):
    result = griewank(X) + ackley(X) + hcf(X)
    return result

def hybrid_composition3(X):
    result = zakharov(X) + michalewicz(X) + levy(X)
    return result

def step(X):
    result = np.sum(np.floor(X + 0.5)**2, axis=1)
    return result

def quartic(X):
    result = np.sum(np.arange(1, X.shape[1] + 1) * X**4, axis=1) + np.random.uniform(0, 1)
    return result

def schaffer2(X):
    result = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        for j in range(X.shape[1] - 1):
            term1 = X[i, j]**2 + X[i, j+1]**2
            sin_term = np.sin(np.sqrt(term1))**2
            denom_term = (1 + 0.001 * term1)**2
            result[i] += 0.5 + (sin_term - 0.5) / denom_term
    return result

def brown(X):
    result = np.sum((X[:, :-1]**2)**(X[:, 1:]**2 + 1) + (X[:, 1:]**2)**(X[:, :-1]**2 + 1), axis=1)
    return result

def exponential(X):
    result = -np.exp(-0.5 * np.sum(X**2, axis=1))
    return result

def styblinski_tang(X):
    result = 0.5 * np.sum(X**4 - 16*X**2 + 5*X, axis=1)
    return result

def sum_squares(X):
    result = np.sum(np.arange(1, X.shape[1] + 1) * X**2, axis=1)
    return result

def goldstein_price(X):
    result = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        x = X[i, 0]
        y = X[i, 1]
        term1 = 1.0 + (x + y + 1.0)**2 * (19.0 - 14.0*x + 3.0*x**2 - 14.0*y + 6.0*x*y + 3.0*y**2)
        term2 = 30.0 + (2.0*x - 3.0*y)**2 * (18.0 - 32.0*x + 12.0*x**2 + 48.0*y - 36.0*x*y + 27.0*y**2)
        result[i] = term1 * term2
    return result

def easom(X):
    result = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        x = X[i, 0]
        y = X[i, 1]
        term1 = -np.cos(x)
        term2 = -np.cos(y)
        term3 = np.exp(-(x - np.pi)**2 - (y - np.pi)**2)
        result[i] = term1 * term2 * term3
    return result

def drop_wave(X):
    result = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        x = X[i, 0]
        y = X[i, 1]
        numerator = 1 + np.cos(12 * np.sqrt(x**2 + y**2))
        denominator = 0.5 * (x**2 + y**2) + 2
        if x == 0 and y == 0:
            result[i] = 0
        else:
            result[i] = - (numerator / denominator)
    return result
