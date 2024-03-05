import os
from glob import glob
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = 'Repository for DW0028: Learned regularizations for elastic full waveform inversion using diffusion models.'

from setuptools import setup

setup(
    name="diffefwi",
    description=descr,
    long_description=open(src('README.md')).read(),
    keywords=['inverse problems',
              'generative models',
              'deep learning',
              'tomography',
              'efwi',
              'seismic'],
    author='Mohammad H. Taufik, Fu Wang, Tariq Alkhalifah',
    author_email='mohammad.taufik@kaust.edu.sa, fu.wang@kaust.edu.sa, tariq.alkhalifah@kaust.edu.sa',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'setuptools_scm',
    ],
)