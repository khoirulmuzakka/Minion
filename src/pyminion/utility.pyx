import numpy as np 
cimport numpy as np

cdef np.ndarray latin_hypercube_sampling(object bounds, int population_size):
    """
    Generate a population using Latin Hypercube Sampling within given bounds.

    Parameters:
    bounds (list of tuple): List of (min, max) pairs for each dimension.
    population_size (int): Number of samples to generate.

    Returns:
    np.ndarray: Array of shape (population_size, len(bounds)) with the generated samples.
    """
    dimensions = len(bounds)
    rng = np.random.default_rng()
    sample = np.empty((population_size, dimensions))

    # Generate quantile levels
    quantiles = rng.uniform(0, 1, size=population_size)

    # Shuffle quantile levels within each column
    for i in range(dimensions):
        quantiles_col = quantiles.copy()
        rng.shuffle(quantiles_col)
        sample[:, i] = bounds[i][0] + quantiles_col * (bounds[i][1] - bounds[i][0])

    return np.array(sample)


cdef tuple getMeanStd(object arr, object weight) : 
    """
    Compute the weighted mean and standard deviation of an array.

    This method calculates the mean and standard deviation of a given array
    of values, taking into account the specified weights for each element.

    Parameters:
    ----------
    arr : array-like
        An array of numerical values for which the mean and standard deviation 
        are to be calculated.
    weight : array-like
        An array of weights corresponding to each element in `arr`. The weights 
        determine the contribution of each element to the mean and standard 
        deviation.

    Returns:
    -------
    tuple
        A tuple containing two values:
        - mean : float
            The weighted mean of the input array.
        - std : float
            The weighted standard deviation of the input array.

    Notes:
    -----
    - The weights are normalized such that they sum to 1 before computing
      the weighted mean and standard deviation.
    - The standard deviation is calculated based on the weighted variance,
      which considers the weights when computing how spread out the values are
      around the mean.
    - Both input arrays (`arr` and `weight`) should be of the same length,
      otherwise a ValueError will be raised by NumPy.
    """

    arr = np.array(arr)
    weight = np.array(weight)
    weight = weight / np.sum(weight)
    mean = np.average(arr, weights=weight)
    variance = np.average((arr - mean)**2, weights=weight)
    std = np.sqrt(variance)
    return mean, std


cdef void enforce_bounds(np.ndarray[np.float64_t, ndim=2] new_candidates, np.ndarray[np.float64_t, ndim=2] bounds, object strategy):
    """
    Handle boundary violations by applying specified strategies to ensure candidate solutions 
    are within defined bounds.

    Parameters:
    ----------
    new_candidates : np.ndarray
        An array of new candidate solutions that need boundary enforcement.
    bounds : np.ndarray
        An array of shape (n_dimensions, 2) representing the lower and upper bounds 
        for each dimension.
    strategy : str
        The strategy to handle boundary violations. Options include:
        - "clip": Clipping values to the bounds.
        - "reflect": Reflecting values back into the bounds.
        - "random": Randomly sampling a new value within the bounds if out of range.
        - "random-leftover" : Randomly sampling a new value within the a new bound defined as the distance between the point and bound. 

    Notes:
    -----
    - Ensure that the `new_candidates` array has the same number of dimensions 
      as specified by the `bounds`.
    - The `bounds` array should have shape (n_dimensions, 2) with the first column 
      representing lower bounds and the second column representing upper bounds.
    """

    cdef int d, i, dim = bounds.shape[0]
    cdef np.float64_t lower_bound, upper_bound, e

    # Loop through each dimension
    for d in range(dim):
        lower_bound = bounds[d, 0]
        upper_bound = bounds[d, 1]
        e = upper_bound - lower_bound  # Range of current dimension

        if strategy == "clip":
            # Clip values that exceed the bounds
            new_candidates[:, d] = np.clip(new_candidates[:, d], lower_bound, upper_bound)

        elif strategy == "reflect":
            # Reflect values back into bounds if they are out of range
            for i in range(new_candidates.shape[0]):
                if new_candidates[i, d] < lower_bound:
                    new_candidates[i, d] = lower_bound + (lower_bound - new_candidates[i, d])
                elif new_candidates[i, d] > upper_bound:
                    new_candidates[i, d] = upper_bound - (new_candidates[i, d] - upper_bound)

        elif strategy == "random":
            # Randomly sample a new value within the bounds for out-of-bounds candidates
            for i in range(new_candidates.shape[0]):
                if new_candidates[i, d] < lower_bound or new_candidates[i, d] > upper_bound:
                    new_candidates[i, d] = np.random.uniform(low=lower_bound, high=upper_bound)

        elif strategy == "random-leftover" : 
            # Resample a new value within a limited range close to the bounds for out-of-range candidates
            for i in range(new_candidates.shape[0]):
                # Check for lower bound violations
                if new_candidates[i, d] < lower_bound:
                    d_lower = abs(new_candidates[i, d] - lower_bound)
                    low_range = lower_bound
                    high_range = lower_bound + min(d_lower, e)
                    new_candidates[i, d] = np.random.uniform(low=low_range, high=high_range)

                # Check for upper bound violations
                elif new_candidates[i, d] > upper_bound:
                    d_upper = abs(new_candidates[i, d] - upper_bound)
                    low_range = upper_bound - min(d_upper, e)
                    high_range = upper_bound
                    new_candidates[i, d] = np.random.uniform(low=low_range, high=high_range)

        else:
            raise ValueError("Invalid strategy. Choose from 'clip', 'reflect', or 'random'.")
