from setuptools import setup

__version__ = '0.0'

with open("README.md") as f:
    README = f.readlines()

#with open("LICENSE") as f:
#    LICENSE = f.readlines()

LICENSE = ''

REQUIREMENTS = ['pandas==0.23.4', 'numpy==1.15.4', 'numba==0.41.0', 'scipy==1.2.0']

setup(
    name='categorical_kneighbors',
    packages=['categorical_kneighbors', ],
    version=__version__,
    install_requires=REQUIREMENTS,
    author="Edward Turner",
    author_email="edward.turnerr@gmail.com",
    description=README,
    license=LICENSE
)