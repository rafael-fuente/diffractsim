from setuptools import setup, find_packages

long_description = """
# Diffractsim: A diffraction simulator for exploring and visualizing physical optics
Flexible, and easy-to-use Python diffraction simulator that focuses on visualizing physical optics phenomena.
The simulator uses mainly scalar diffraction techniques for light propagation, provides an interface for simulation set up, and includes several plotting options, counting with CIE Color matching functions for accurate color reproduction.

The basic use of this simulator using the angular spectrum method is described in this [article](https://rafael-fuente.github.io/simulating-diffraction-patterns-with-the-angular-spectrum-method-and-python.html).
Take a look to the [videos](https://www.youtube.com/watch?v=Ft8CMEooBAE&list=PLYkZehxPE_IhyO6wC21nFP0q1ZYGIW4l1&index=1) to see the animated simulations!
"""


setup(
    name='diffractsim',
    version='2.2.3',
    description='A diffraction simulator for exploring and visualizing physical optics',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method',
    download_url='https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/archive/main.zip',
    keywords = ['diffraction', 'angular spectrum method', 'optics', 'physics simulation'],
    author='Rafael de la Fuente',
    author_email='rafael.fuente.herrezuelo@gmail.com',
    license='MPL 2.0',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'Pillow', 'matplotlib', 'progressbar', 'autograd'],
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