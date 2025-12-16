from .util.backend_functions import get_backend, set_backend
from .util.backend_functions import backend as bd
from .util.backend_functions import backend as bd
from .util.image_handling import load_image_as_function
from .util.file_handling import load_file_as_function, load_phase_as_function
from .polychromatic_simulator import PolychromaticField
from .monochromatic_simulator import MonochromaticField
from . import colour_functions as cf
from .polynomials import zernike_polynomial
from .holography import FourierPhaseRetrieval, CustomPhaseRetrieval, RotationalPhaseDesign
from .diffractive_elements import *
from .light_sources import *

# Vectorial field and TMM modules
from .vectorial_field import VectorialField
from .tmm import (
    Layer, Stack, fresnel_coefficients, Material, DrudeMaterial,
    IdealPolarizer, IdealRetarder, quarter_wave_plate, half_wave_plate,
    spp_dispersion_relation, spp_effective_index, kretschmann_configuration, single_interface_spp
)

from .util.constants import *
