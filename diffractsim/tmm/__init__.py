"""
Transfer Matrix Method (TMM) for planar multilayer stacks.

MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

from .layer import Layer
from .stack import Stack
from .fresnel import fresnel_coefficients
from .materials import Material, DrudeMaterial
from .thin_elements import IdealPolarizer, IdealRetarder, quarter_wave_plate, half_wave_plate
from .spp import spp_dispersion_relation, spp_effective_index, kretschmann_configuration, single_interface_spp

__all__ = [
    'Layer', 'Stack', 'fresnel_coefficients', 'Material', 'DrudeMaterial',
    'IdealPolarizer', 'IdealRetarder', 'quarter_wave_plate', 'half_wave_plate',
    'spp_dispersion_relation', 'spp_effective_index', 'kretschmann_configuration', 'single_interface_spp'
]

