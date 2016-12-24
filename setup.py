try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

config = {
    'name': 'adascreen',
    'description': 'AdaScreen - Ensembles of Screening Rules',
    'url': 'https://github.com/nicococo/AdaScreen',
    'author': 'Nico Goernitz',
    'author_email': 'nico.goernitz@tu-berlin.de',
    'version': '2016.12.21',
    'install_requires': ['nose', 'scikit-learn', 'numpy', 'multiprocessing',
                         'drmaa', 'psutil', 'pyzmq'],
    'packages': ['adascreen', 'adascreen_experiments', 'clustermap'],  # names of the packages
    'package_dir': {'adascreen' : 'adascreen',
                    'clustermap' : 'clustermap',
                    'adascreen_experiments' : 'scripts'},  # locations of the actual package in the source tree
    'scripts': ['bin/adascreen_experiment.sh'],
    'ext_modules' : cythonize([Extension("enet_solver",["adascreen/enet_solver.pyx"],
        include_dirs=[np.get_include()])]),
    'classifiers':['Intended Audience :: Science/Research',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7']
}

setup(**config)
