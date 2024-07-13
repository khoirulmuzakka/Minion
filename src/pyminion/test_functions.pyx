import numpy as np 
cimport numpy as np 

cpdef np.ndarray[np.float64_t, ndim=1] sphere(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = np.sum(X**2, axis=1)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] rosenbrock(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = np.sum(100.0 * (X[:, 1:] - X[:, :-1]**2.0)**2.0 + (1 - X[:, :-1])**2.0, axis=1)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] rastrigin(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = np.sum(X**2 - 10.0 * np.cos(2.0 * np.pi * X) + 10, axis=1)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] griewank(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = np.sum(X**2, axis=1) / 4000.0 - np.prod(np.cos(X / np.sqrt(np.arange(1, X.shape[1] + 1))), axis=1) + 1
    return result

cpdef np.ndarray[np.float64_t, ndim=1] ackley(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(X**2, axis=1) / X.shape[1])) - \
             np.exp(np.sum(np.cos(2.0 * np.pi * X), axis=1) / X.shape[1]) + 20 + np.e
    return result

cpdef np.ndarray[np.float64_t, ndim=1] zakharov(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = np.sum(X**2, axis=1) + np.sum(0.5 * np.arange(1, X.shape[1] + 1) * X, axis=1)**2 + \
             np.sum(0.5 * np.arange(1, X.shape[1] + 1) * X, axis=1)**4
    return result

cpdef np.ndarray[np.float64_t, ndim=1] michalewicz(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = -np.sum(np.sin(X) * (np.sin(np.arange(1, X.shape[1] + 1) * X**2 / np.pi))**(2 * 10), axis=1)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] levy(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] term1, term2, term3, result
    cdef np.ndarray[np.float64_t, ndim=2] w
    w = 1 + (X - 1) / 4
    term1 = (np.sin(np.pi * w[:, 0]))**2
    term2 = np.sum((w[:, :-1] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[:, :-1] + 1))**2), axis=1)
    term3 = (w[:, -1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[:, -1]))**2)
    result = term1 + term2 + term3
    return result

cpdef np.ndarray[np.float64_t, ndim=1] dixon_price(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = (X[:, 0] - 1)**2 + np.sum(np.arange(2, X.shape[1] + 1) * (2 * X[:, 1:]**2 - X[:, :-1])**2, axis=1)
    return result

# Define additional test functions

cpdef np.ndarray[np.float64_t, ndim=1] bent_cigar(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = X[:, 0]**2 + 1e6 * np.sum(X[:, 1:]**2, axis=1)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] discus(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = 1e6 * X[:, 0]**2 + np.sum(X[:, 1:]**2, axis=1)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] weierstrass(np.ndarray[np.float64_t, ndim=2] X):
    cdef float a = 0.5
    cdef int b = 3, k_max = 20
    cdef int n = X.shape[1]
    cdef int i, k

    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(X.shape[0], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] inner_sum = np.zeros_like(X)
    
    for i in range(n):
        for k in range(k_max):
            inner_sum[:, i] += a**k * np.cos(2 * np.pi * b**k * (X[:, i] + 0.5))
    result = np.sum(inner_sum, axis=1) - n * np.sum([a**k * np.cos(np.pi * b**k) for k in range(k_max)]) 
    return result

cpdef np.ndarray[np.float64_t, ndim=1] happy_cat(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = ((np.sum(X**2, axis=1) - X.shape[1])**2)**0.25 + (0.5 * np.sum(X**2, axis=1) + np.sum(X, axis=1)) / X.shape[1] + 0.5
    return result

cpdef np.ndarray[np.float64_t, ndim=1] hgbat(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = ((np.sum(X**2, axis=1)**2)**0.25 + (0.5 * np.sum(X**2, axis=1) + np.sum(X, axis=1)) / X.shape[1] + 0.5)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] hcf(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = np.sum(np.abs(X), axis=1) * np.exp(np.sum(np.abs(X), axis=1) / X.shape[1])
    return result

cpdef np.ndarray[np.float64_t, ndim=1] grie_rosen(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = 1 + np.sum(100 * (X[:, 1:] - X[:, :-1]**2)**2 + (1 - X[:, :-1])**2, axis=1)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] escaffer6(np.ndarray[np.float64_t, ndim=2] X):
    cdef int n = X.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i, j
    cdef np.float64_t term1, sin_term, denom_term

    for i in range(n):
        for j in range(X.shape[1] - 1):
            term1 = X[i, j]**2 + X[i, j+1]**2
            sin_term = np.sin(np.sqrt(term1))**2
            denom_term = (1 + 0.001 * term1)**2
            result[i] += 0.5 + (sin_term - 0.5) / denom_term
    return result

cpdef np.ndarray[np.float64_t, ndim=1] hybrid_composition1(np.ndarray[np.float64_t, ndim=2] X):
    # Assuming HC1 is a combination of multiple functions
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = np.sum([
        sphere(X),
        rosenbrock(X),
        rastrigin(X)
    ], axis=0)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] hybrid_composition2(np.ndarray[np.float64_t, ndim=2] X):
    # Assuming HC2 is a combination of multiple functions
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = np.sum([
        griewank(X),
        ackley(X),
        hcf(X)
    ], axis=0)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] hybrid_composition3(np.ndarray[np.float64_t, ndim=2] X):
    # Assuming HC3 is a combination of multiple functions
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = np.sum([
        zakharov(X),
        michalewicz(X),
        levy(X)
    ], axis=0)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] step(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = np.sum(np.floor(X + 0.5)**2, axis=1)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] quartic(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = np.sum(np.arange(1, X.shape[1] + 1) * X**4, axis=1) + np.random.uniform(0, 1)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] schaffer2(np.ndarray[np.float64_t, ndim=2] X):
    cdef int n = X.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i, j
    cdef np.float64_t term1, sin_term, denom_term

    for i in range(n):
        for j in range(X.shape[1] - 1):
            term1 = X[i, j]**2 + X[i, j+1]**2
            sin_term = np.sin(np.sqrt(term1))**2
            denom_term = (1 + 0.001 * term1)**2
            result[i] += 0.5 + (sin_term - 0.5) / denom_term
    return result

cpdef np.ndarray[np.float64_t, ndim=1] brown(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = np.sum((X[:, :-1]**2)**(X[:, 1:]**2 + 1) + (X[:, 1:]**2)**(X[:, :-1]**2 + 1), axis=1)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] exponential(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = -np.exp(-0.5 * np.sum(X**2, axis=1))
    return result

cpdef np.ndarray[np.float64_t, ndim=1] styblinski_tang(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = 0.5 * np.sum(X**4 - 16*X**2 + 5*X, axis=1)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] sum_squares(np.ndarray[np.float64_t, ndim=2] X):
    cdef np.ndarray[np.float64_t, ndim=1] result
    result = np.sum(np.arange(1, X.shape[1] + 1) * X**2, axis=1)
    return result

cpdef np.ndarray[np.float64_t, ndim=1] goldstein_price(np.ndarray[np.float64_t, ndim=2] X):
    cdef int n = X.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i
    cdef np.float64_t x, y, term1, term2, term3, term4, term5, term6, term7, term8

    for i in range(n):
        x = X[i, 0]
        y = X[i, 1]
        
        term1 = 1.0 + (x + y + 1.0)**2 * (19.0 - 14.0*x + 3.0*x**2 - 14.0*y + 6.0*x*y + 3.0*y**2)
        term2 = 30.0 + (2.0*x - 3.0*y)**2 * (18.0 - 32.0*x + 12.0*x**2 + 48.0*y - 36.0*x*y + 27.0*y**2)
        
        result[i] = term1 * term2

    return result

cpdef np.ndarray[np.float64_t, ndim=1] easom(np.ndarray[np.float64_t, ndim=2] X):
    cdef int n = X.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i
    cdef double x, y, term1, term2, term3

    for i in range(n):
        x = X[i, 0]
        y = X[i, 1]
        
        term1 = -np.cos(x)
        term2 = -np.cos(y)
        term3 = np.exp(-(x - np.pi)**2 - (y - np.pi)**2)
        
        result[i] = term1 * term2 * term3

    return result

cpdef np.ndarray[np.float64_t, ndim=1] drop_wave(np.ndarray[np.float64_t, ndim=2] X):
    cdef int n = X.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i
    cdef double x, y, numerator, denominator

    for i in range(n):
        x = X[i, 0]
        y = X[i, 1]
        
        numerator = 1 + np.cos(12 * np.sqrt(x**2 + y**2))
        denominator = 0.5 * (x**2 + y**2) + 2
        
        if x == 0 and y == 0:
            result[i] = 0  # handle division by zero at (0, 0)
        else:
            result[i] = - (numerator / denominator)
    
    return result
