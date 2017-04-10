from distutils.core import setup, Extension
import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

objective = Extension('_objective',
                      sources=['pyorderedfuzzy/ofmodels/src/objective_wrap.c',
                               'pyorderedfuzzy/ofmodels/src/objective.c',
                               'pyorderedfuzzy/ofmodels/src/utils.c'],
                      include_dirs=[numpy_include],)

setup(name='objective',
      description="""c extensions""",
      author="amarszalek",
      version="1.0",
      ext_modules=[objective],
      py_modules=["../objective"],)
