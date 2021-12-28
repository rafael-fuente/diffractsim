from setuptools import setup, find_packages

long_description = """
# Diffractsim: A diffraction simulator for exploring and visualizing physical optics
Implementation of the angular spectrum method as well as other light propagation methods in Python to simulate diffraction patterns with arbitrary apertures. You can use it for simulating both monochromatic and polychromatic light also with arbitrary spectrums.

How the method and the simulator works is described in this [Article](https://rafael-fuente.github.io/simulating-diffraction-patterns-with-the-angular-spectrum-method-and-python.html). Take a look to the [video](https://youtu.be/Ft8CMEooBAE) to see the animated simulations!
"""


setup(
    name='diffractsim',
    version='2.0.0',
    description='A diffraction simulator for exploring and visualizing physical optics',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method',
    download_url='https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/archive/main.zip',
    keywords = ['diffraction', 'angular spectrum method', 'optics', 'physics simulation'],
    author='Rafael de la Fuente',
    author_email='rafael.fuente.herrezuelo@gmail.com',
    license='BSD-3-Clause',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'Pillow', 'matplotlib', 'progressbar'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',

    ],
    include_package_data = True,
    python_requires ='>=3.6',
)