from setuptools import setup, find_packages

setup(
    name='spuq',
    version='0.1',
    description='Spectral Methods for Uncertainty Quantification',
    author='SpuqDevs',
    author_email='e.zander@tu-bs.de',
    url='http://github.com/SpuqTeam/spuq',
    packages=find_packages(),
    long_description="""\
      spuq is a library for performing uncertainty quantification using
      spectral methods in python
      """,
    classifiers=[
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        ],
    keywords='uncertainty quantification spectral method stochastic finite elements',
    license='GPL',
    install_requires=[
        'setuptools',
        'scipy',
        'numpy',
        ],
    )
