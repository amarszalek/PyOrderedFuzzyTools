from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(
    ext_modules=[Extension("_objective", ["pyorderedfuzzy/ofmodels/src/_objective.c",
                                          "pyorderedfuzzy/ofmodels/src/objective.c",
                                          "pyorderedfuzzy/ofmodels/src/utils.c"])],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
