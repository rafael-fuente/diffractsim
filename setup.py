from setuptools import setup

long_description = """
# Diffraction Simulations - Angular Spectrum Method
Implementation of the angular spectrum method in Python to simulate diffraction patterns with arbitrary apertures. You can use it for simulating both monochromatic and polychromatic light also with arbitrary spectrums.

How the method and the simulator work is described in this [Article](https://rafael-fuente.github.io/simulating-diffraction-patterns-with-the-angular-spectrum-method-and-python.html). Take a look to the [Youtube video](https://youtu.be/Ft8CMEooBAE) to see the animated simulations!
"""


setup(
    name='diffractsim',
    version='1.2.0',
    description='Implementation of the angular spectrum method in Python to simulate diffraction patterns',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method',
    download_url='https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/archive/main.zip',
    keywords = ['diffraction', 'angular spectrum method', 'optics', 'physics simulation'],
    author='Rafael de la Fuente',
    author_email='rafael.fuente.herrezuelo@gmail.com',
    license='BSD-3-Clause',
    packages=['diffractsim'],
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