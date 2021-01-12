from setuptools import setup

with open("readme.md", "r") as file:
    long_description = file.read()


setup(
    name='diffractsim',
    version='1.1.0',
    description='Implementation of the Angular Spectrum method in Python to simulate Diffraction Patterns',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method',
    download_url='https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/archive/main.zip',
    keywords = ['diffraction', 'angular spectrum method', 'optics', 'physics simulation'],
    author='Rafael de la Fuente',
    author_email='rafael.fuente.herrezuelo@gmail.com',
    license='MIT',
    packages=['diffractsim'],
    install_requires=['numpy', 'scipy', 'Pillow', 'matplotlib', 'progressbar'],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',

    ],
    include_package_data = True,
    python_requires ='>=3.6',
)