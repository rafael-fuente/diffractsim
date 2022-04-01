# Diffraction Simulations with Lenses

These examples correspond to the simulations with lenses, whose explanation can be found in [this video](https://www.youtube.com/watch?v=G4J4PV6tqH0).
A further discussion and its mathematical background can be found in this [article](https://rafael-fuente.github.io/simulating-light-diffraction-with-lenses-visualizing-fourier-optics.html).

## Installation

```
pip install diffractsim
```

Alternatively, to download the examples and the apertures as well, you can also build from source by cloning the repository and running from the main folder project on the command prompt:
```
python setup.py install
```

## Simulations

To perform the simulations, just run from the [examples subdirectory](https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/tree/main/examples) the corresponding Python scripts on the command prompt:


```
python bahtinov_mask.py
```

[![animation](/images/bahtinov_mask.gif)](https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/blob/main/examples/bahtinov_mask.py)

```
python optical_imaging_system.py
```

[![animation](/images/optical_imaging_system.gif)](https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/blob/main/examples/optical_imaging_system.py)

```
python object_behind_the_lens.py
```

[![animation](/images/object_behind_the_lens.gif)](https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/blob/main/examples/object_behind_the_lens.py)

```
python spatial_filter.py
```

[![animation](/images/spatial_filter.gif)](https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/blob/main/examples/spatial_filter.py)

```
python hexagonal_aperture_lens.py
```

[![animation](/images/hexagon_with_lens.gif)](https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/blob/main/examples/hexagonal_aperture_lens.py)

```
python circular_aperture_lens.py
```

[![animation](/images/circular_aperture_lens.gif)](https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/blob/main/examples/circular_aperture_lens.py)

```
python beyond_the_focal_length.py
```

[![animation](/images/beyond_the_focal_length.gif)](https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/blob/main/examples/beyond_the_focal_length.py)

The scripts uploaded only render the diffraction pattern at a single screen distance. If you want to create an animation, you must loop through the different distances and then merging the rendered frames to a video, for example, using FFmpeg.
