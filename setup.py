try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'AdaScreen - Ensembles of Screening Rules',
    'url': 'https://github.com/nicococo/AdaScreen',
    'author': 'Nico Goernitz',
    'author_email': 'nico.goernitz@tu-berlin.de',
    'version': '2016.12.21',
    'install_requires': ['nose', 'scikit-learn', 'numpy', 'gridmap'],
    'packages': ['adascreen', 'adascreen_experiments'],  # names of the packages
    'package_dir': {'adascreen' : 'adascreen', 'adascreen_experiments' : 'scripts'},  # locations of the actual package in the source tree
    'scripts': ['bin/adascreen_experiment.sh'],
    'name': 'adascreen',
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