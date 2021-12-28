from .backend_functions import backend as bd


def scaled_fourier_transform(x, y, U, λ = 1,z =1, scale_factor = 1, mesh = False):
    """ 

    Computes de following integral:

    Uf(x,y) = ∫∫  U(u, v) * exp(-1j*pi/ (z*λ) *(u*x + v*y)) * du*dv

    Given the extent of the input coordinates of (u, v) of U(u, v): extent_u and extent_v respectively,
    Uf(x,y) is evaluated in a scaled coordinate system (x, y) with:

    extent_x = scale_factor*extent_u
    extent_y = scale_factor*extent_v

    """
    global bd
    from .backend_functions import backend as bd

    Ny,Nx = U.shape    
    
    if mesh == False:
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        xx, yy = bd.meshgrid(x, y)
    else:
        dx = x[0,1]-x[0,0]
        dy = y[1,0]-y[0,0]
        xx, yy = x,y

    extent_x = dx*Nx
    extent_y = dy*Ny

    L1 = extent_x
    L2 = extent_x*scale_factor

    f_factor = 1/(λ*z)
    fft_U = bd.fft.fftshift(bd.fft.fft2(U * bd.exp(-1j*bd.pi* f_factor*(xx**2 + yy**2) ) * bd.exp(1j*bd.pi*(L1- L2)/L1 * f_factor*(xx**2 + yy**2 ))))
    
    
    fx = bd.fft.fftshift(bd.fft.fftfreq(Nx, d = dx))
    fy = bd.fft.fftshift(bd.fft.fftfreq(Ny, d = dy))
    fxx, fyy = bd.meshgrid(fx, fy)

    Uf = bd.fft.ifft2(bd.fft.ifftshift( bd.exp(- 1j * bd.pi / f_factor * L1/L2 * (fxx**2 + fyy**2))  *  fft_U) )
    
    extent_x = extent_x*scale_factor
    extent_y = extent_y*scale_factor

    dx = dx*scale_factor
    dy = dy*scale_factor

    x = x*scale_factor
    y = y*scale_factor

    xx = xx*scale_factor
    yy = yy*scale_factor  

    Uf = L1/L2 * bd.exp(-1j *bd.pi*f_factor* (xx**2 + yy**2)   - 1j * bd.pi*f_factor* (L1-L2)/L2 * (xx**2 + yy**2)) * Uf *1j * (λ*z)

    if mesh == False:
        return x, y, Uf
    else:
        return xx, yy, Uf



