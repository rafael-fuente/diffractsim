# Diffractsim: A diffraction simulator for exploring and visualizing physical optics
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6147772.svg)](https://doi.org/10.5281/zenodo.6147772)

[![animation](/images/diffraction_animated.gif)](https://www.youtube.com/watch?v=Ft8CMEooBAE&list=PLYkZehxPE_IhyO6wC21nFP0q1ZYGIW4l1&index=1)


Flexible, and easy-to-use Python diffraction simulator that focuses on visualizing physical optics phenomena. The simulator uses mainly scalar diffraction techniques for light propagation, provides a nice interface for simulation set up, and includes several plotting options, counting with CIE Color matching functions for accurate color reproduction.

The basic use of this simulator using the angular spectrum method is described in this [article](https://rafael-fuente.github.io/simulating-diffraction-patterns-with-the-angular-spectrum-method-and-python.html). Take a look to the [videos](https://www.youtube.com/watch?v=Ft8CMEooBAE&list=PLYkZehxPE_IhyO6wC21nFP0q1ZYGIW4l1&index=1) to see the animated simulations!

## Features

- [x] Arbitrary apertures and light spectrums
- [x] Full-path optical propagation and arbitrary zoom in the region of interest
- [x] Lenses
- [x] Phase holograms generation and reconstruction
- [x] GPU acceleration
- [ ] Incoherent Light (coming soon)


## Installation
```
pip install diffractsim
```

Alternatively, to download the examples and the apertures as well, you can also build from source by cloning the repository and running from the main folder project on the command prompt:
```
python setup.py install
```

## Examples

To perform the simulations, just run from the [examples subdirectory](https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/tree/main/examples) the corresponding Python scripts on the command prompt. 
To compute your own diffraction pattern, you'll need to specify in the script the aperture as an image and input its size.

```
python hexagon_monochromatic.py
```

[![N|Solid](/images/hexagon_monochromatic.png)](/examples/hexagon_monochromatic.py)

```
python hexagon_polychromatic.py
```

[![N|Solid](/images/hexagon_polychromatic.png)](/examples/hexagon_polychromatic.py)

```
python rectangular_grating_small.py
```

[![N|Solid](/images/rectangular_grating_small.png)](/examples/rectangular_grating_small.py)

```
python rectangular_grating_big.py
```

[![N|Solid](/images/rectangular_grating_big.png)](/examples/rectangular_grating_big.py)

```
python bahtinov_mask.py
```

[![N|Solid](/images/bahtinov_mask.png)](/examples/bahtinov_mask.py)

```
python rings.py
```

[![N|Solid](/images/rings.png)](/examples/rings.py)

```
python hexagonal_grating.py
```

[![N|Solid](/images/hexagonal_grating.png)](/examples/hexagonal_grating.py)

```
python diffraction_text.py
```

[![N|Solid](/images/text.png)](/examples/text.py)

The examples from the video about diffraction with lenses can be found [here](https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/blob/main/Simulations%20with%20lenses.md).

GPU acceleration requires having [CuPy](https://docs.cupy.dev/en/stable/install.html) installed and [CUDA](https://developer.nvidia.com/cuda-downloads) in your computer. 
To use GPU acceleration in your simulations, after import `diffractsim` add the line:

```python
diffractsim.set_backend("CUDA")
```
Cupy and CUDA aren't required to install and use this package, but they can offer a significant speed boost.

The first GPU accelerated run can be slow because Python is caching the required functions. The next time it can be about 10x and 100x faster than a CPU backend depending on your GPU. The speed boost raises as the grid gets larger.


Diffractsim can also be used to compute and visualize longitudinal profiles. Since the computation of each propagation distance is independent, it can be fully parallelized, and therefore GPU use is highly recommended.

```
python lens_longitudinal_profile.py
```

[![N|Solid](/images/lens_longitudinal_profile.png)](/examples/lens_longitudinal_profile.py)


```
python axicon_longitudinal_profile.py
```

[![N|Solid](/images/axicon_longitudinal_profile.png)](/examples/axicon_longitudinal_profile.py)


The problem of phase retrieval is a classic one in optics and arises when one is interested in retrieving the wavefront from two intensity measurements acquired in two different planes. Diffractsim provides a simple implementation of this problem.

In the following example, the GitHub logo is recovered at the Fourier plane from a coherently illuminated square-shaped aperture. The script generates a phase mask, which is stored as an image using an HSV colormap and then placed on the aperture to reconstruct the desired image.

```
python phase_hologram_github_logo_generation_and_reconstruction.py
```

[![animation](/images/github_logo.gif)](/examples/phase_hologram_github_logo_generation_and_reconstruction.py)
