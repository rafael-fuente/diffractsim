from scipy.special import assoc_laguerre, gamma
from ..util.backend_functions import backend as bd

def laguerre_gaussian_mode(p,l,xx,yy,w0):
    """
    Evaluates the Laguerre Gaussian mode n,m as a function of x and y

    Parameters
    ----------
    p: positive integer
    l: positive integer
    xx: 2D numpy array
    yy: 2D numpy array
    w0 : beam waist
    """

    global bd
    from ..util.backend_functions import backend as bd

    r = bd.sqrt(xx**2 + yy**2)
    phi = bd.arctan2(yy, xx)
    return 1/(bd.sqrt(gamma(l+1))) * (bd.sqrt(2)*r/w0)**l * assoc_laguerre(2*(r/w0)**2, p, l) * bd.exp(1j * phi * l) * bd.exp( - (r/w0)**2)
