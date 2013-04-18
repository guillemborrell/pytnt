from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext



setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("_refine_fast",
                             ["_refine_point_list.pyx"],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp'],)],
    include_dirs = ['/opt/epd-7.3-1-rh5-x86_64/lib/python2.7/site-packages/numpy/core/include']
    )
