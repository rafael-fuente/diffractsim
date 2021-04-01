import numpy

try:
    import cupy
    CUPY_CUDA_AVAILABLE = True
except ImportError:
    CUPY_CUDA_AVAILABLE = False

global backend
backend = numpy

def set_backend(name: str):
    """ Set the backend for the simulations
    This way, all methods of the backend object will be replaced.
    Args:
        name: name of the backend. Allowed backend names:
            - ``CPU``
            - ``CUDA``
    """
    # perform checks
    if name == "CUDA" and not CUPY_CUDA_AVAILABLE:
        raise RuntimeError(
            "Cupy CUDA backend is not available.\n"
            "Do you have a GPU on your computer?\n"
            "Is Cupy with CUDA support installed?"
        )
    global backend
    # change backend
    if name == "CPU":
        backend = numpy
    elif name == "CUDA":
        backend = cupy
    else:
        raise RuntimeError(f'unknown backend "{name}"')


def get_backend():
    global backend    
    print(backend)




