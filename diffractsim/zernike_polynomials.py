from scipy.special import gamma
from .backend_functions import backend as bd

def zernike_polynomial(n,m, x, y):
    """
    Evaluates the Zernike Polynomial n,m as a function of x and y over the region sqrt(x**2 + y**2) <= 1
    Can be used for modeling aberrations in add_lens method

    Parameters
    ----------
    n: integer
    m: integer
    x: 2D numpy array
    y: 2D numpy array
    """

    global bd
    from .backend_functions import backend as bd

    𝜃 = bd.arctan2(y,x)
    r = bd.sqrt(x*x + y*y)


    R_m__n=0

    for s in range(0, (n-abs(m))//2 + 1):
        R_m__n += (-1)**s * (gamma(n-s +1) /(gamma(s +1) * gamma((n+m)//2 - s +1)  * gamma((n-m)/2 - s +1) )) * (r) **(n-2*s)

    G__m  = bd.cos(m*𝜃) if m>=0 else bd.sin(-m*𝜃)

    R_m__n = bd.where((r) <= 1 , R_m__n, 0 )


    Z_m__n = ((bd.sqrt(2 * (n+1) ) * R_m__n * G__m)  if m != 0  else bd.sqrt((n+1) )*R_m__n  )
    return Z_m__n
