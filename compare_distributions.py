from numpy import log2, sum, isnan
import numpy as np

from scipy.linalg import norm


def KullbackLeibler(x, y):
    d1 = x*log2(2*x/(x+y))
    d1[isnan(d1)] = 0
    d = 0.5*sum(d1)
    return d


def JensenShannon(x,y):
    import warnings
    warnings.filterwarnings("ignore", category = RuntimeWarning)
    d1 = x*log2(2*x/(x+y))
    d2 = y*log2(2*y/(x+y))
    d1[isnan(d1)] = 0
    d2[isnan(d2)] = 0
    d = 0.5*sum(d1+d2)
    return d

def Hellinger(x,y):
    return norm(np.sqrt(x) - np.sqrt(y)) / np.sqrt(2)

