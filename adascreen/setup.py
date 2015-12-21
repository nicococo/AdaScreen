from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

INCLUDE_BLAS = '/usr/include/atlas'
LIB_BLAS = '/usr/lib/atlas-base/atlas'
LIBS = 'blas'

# for windows
#INCLUDE_BLAS = 'C:\OpenBLAS\include'
#LIB_BLAS = 'C:\OpenBLAS\lib'
#LIBS = 'libopenblas'

# needs 
# - numpy/arrayobject...
# - cblas.h 
# 

# setup(
#     #ext_modules=cythonize([Extension("enet_solver",["enet_solver.pyx"],
#     #    include_dirs=[np.get_include(), 'C:\Program Files (x86)\Intel\Composer XE\mkl\include'],
#     #    library_dirs=['C:\Program Files (x86)\Intel\Composer XE\mkl\lib\intel64'],
#     #    extra_compiler_args=['-DMKL_ILP64'],
#     #    libraries=["mkl_intel_ilp64", "mkl_core", "mkl_sequential"])
#     #   ])
#     #ext_modules=cythonize([Extension("enet_solver",["enet_solver.pyx"],
#     #    include_dirs=[np.get_include(), 'C:\mkl\include'],
#     #    library_dirs=['C:\mkl\lib\intel64'],
#     #    extra_compiler_args=['-DMKL_ILP64'],
#     #    libraries=["mkl_intel_ilp64", "mkl_core", "mkl_sequential"])
#     #   ])
#     #ext_modules=cythonize([Extension("enet_solver",["enet_solver.pyx"],
#     #    include_dirs=[np.get_include(), 'C:\mkl\include'])
#     #   ])
#     ext_modules=cythonize([Extension("enet_solver",["enet_solver.pyx"],
#         include_dirs=[np.get_include(), 'C:\OpenBLAS\include'],
#         library_dirs=['C:\OpenBLAS\lib'],
#         libraries=["libopenblas"])
#        ])
# )    

setup(
    ext_modules=cythonize([Extension("enet_solver",["enet_solver.pyx"],
        include_dirs=[np.get_include(), INCLUDE_BLAS],
        library_dirs=[LIB_BLAS],
        libraries=[LIBS])
       ])
)    
