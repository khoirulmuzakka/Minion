from scipy.stats import mannwhitneyu
import numpy as np


def MWUT (A, B, alpha=0.05) : 
    """
    Perform the Mann-Whitney U Test (Wilcoxon rank-sum test) to compare two independent samples.

    This function tests whether the distributions of two independent samples A and B are significantly different. 
    It returns a value indicating whether sample A is significantly higher, lower, or not significantly different from sample B.

    @param A: array-like
        The first independent sample to compare.
    @param B: array-like
        The second independent sample to compare.
    @param alpha: float, optional
        The significance level for the test. Default is 0.05.
        
    @return: int
        Returns 1 if sample A is significantly lower than sample B,
        Returns -1 if sample A is significantly higher than sample B,
        Returns 0 if there is no significant difference between sample A and sample B.
    
    @note
        The test is performed using a two-sided alternative hypothesis.
    """

    statistic, p_value = mannwhitneyu(A, B, alternative='two-sided')
    #print(p_value)
    mA = np.mean(A)
    mB = np.mean(B)

    ret = 0 # null hypothesis
    if p_value <alpha : 
        if mA < mB : ret=1 # A wins
        elif mA >  mB : ret =-1 # A lose
    return ret