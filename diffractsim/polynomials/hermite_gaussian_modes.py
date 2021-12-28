from scipy.special import eval_hermite
from ..util.backend_functions import backend as bd

def hermite_gaussian_mode(n,m, xx, yy, w0):
    """
    Evaluates the Hermite Gaussian mode n,m as a function of x and y

    Parameters
    ----------
    n: positive integer
    m: positive integer
    xx: 2D numpy array
    yy: 2D numpy array
    w0 : beam waist
    """

    global bd
    from ..util.backend_functions import backend as bd

    return eval_hermite(n,bd.sqrt(2) * xx/w0) *eval_hermite(m,bd.sqrt(2) * yy/w0) * bd.exp( - (xx**2+yy**2)/(w0**2))
