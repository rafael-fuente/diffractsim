from .backend_functions import backend as bd
from .backend_functions import backend_name

def chirpz(x, A, W, M):
    """
    
    Parameters
    ----------

    x: array to evaluate chirp-z transform (along last dimension of array)
    A: starting point of chirp-z contour
    W: controls frequency sample spacing and shape of the contour
    M: number of frequency sample points

    Reference:
    Rabiner, L.R., R.W. Schafer and C.M. Rader. The Chirp z-Transform
    Algorithm. IEEE Transactions on Audio and Electroacoustics,
    AU-17(2):86--92, 1969

    Originally Written by Stefan van der Walt: 
    http://www.mail-archive.com/numpy-discussion@scipy.org/msg01812.html
    
    The discrete z-transform,
    X(z) = \sum_{n=0}^{N-1} x_n z^{-n}
    is calculated at M points,
    z_k = AW^-k, k = 0,1,...,M-1
    for A and W complex, which gives
    X(z_k) = \sum_{n=0}^{N-1} x_n z_k^{-n}
    """
    global bd
    global backend_name

    from .backend_functions import backend as bd
    from .backend_functions import backend_name

    if backend_name == 'jax': 
        import jax
        if jax.config.jax_enable_x64:
            complex_ = bd.complex128
        else:
            complex_ = bd.complex64
    else:
        complex_ = complex

    x = bd.asarray(x, dtype=complex_)
    P = x.shape

    N = P[-1]
    L = int(2 ** bd.ceil(bd.log2(M + N - 1)))

    n = bd.arange(N, dtype=float)
    y = bd.power(A, -n) * bd.power(W, n ** 2 / 2.)
    y = bd.tile(y, (P[0], 1)) * x
    Y = bd.fft.fft(y, L)

    n = bd.arange(L, dtype=float)
    v = bd.zeros(L, dtype=complex_)
    if backend_name == 'jax':
        v = v.at[:M].set(bd.power(W, -n[:M] ** 2 / 2.))
        v = v.at[L-N+1:].set(bd.power(W, -(L - n[L-N+1:]) ** 2 / 2.))
    else:
        v[:M] = bd.power(W, -n[:M] ** 2 / 2.)
        v[L-N+1:] = bd.power(W, -(L - n[L-N+1:]) ** 2 / 2.)

    V = bd.fft.fft(v)

    g = bd.fft.ifft(bd.tile(V, (P[0], 1)) * Y)[:,:M]
    k = bd.arange(M)
    g = g * bd.tile(bd.power(W, k ** 2 / 2.), (P[0],1))

    # Return result
    return g
