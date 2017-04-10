from distutils.core import setup, Extension
import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

objective = Extension('_objective',
                      sources=['pyorderedfuzzy/ofmodels/src/objective.i',
                               'pyorderedfuzzy/ofmodels/src/objective.c',
                               'pyorderedfuzzy/ofmodels/src/utils.c'],
                      include_dirs=[numpy_include],
                      swig_opts=['-py3', '-modern', '-I../include'])

setup(
    name='pyorderedfuzzy',
    version='0.0.1',
    packages=['pyorderedfuzzy', 'pyorderedfuzzy.ofnumbers', 'pyorderedfuzzy.ofcandles',
              'pyorderedfuzzy.ofrandoms', 'pyorderedfuzzy.ofmodels'],

    ext_modules=[objective, ],
    py_modules=['objective', ],
    url='',
    license='',
    author='amarszalek',
    author_email='amarszalek@pk.edu.pl',
    description='Python package for Ordered Fuzzy Numbers'
)
