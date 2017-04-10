from distutils.core import setup
import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

setup(
    name='pyorderedfuzzy',
    version='0.0.1',
    packages=['pyorderedfuzzy', 'pyorderedfuzzy.ofnumbers', 'pyorderedfuzzy.ofcandles',
              'pyorderedfuzzy.ofrandoms', 'pyorderedfuzzy.ofmodels'],
    package_data={'pyorderedfuzzy.ofmodels': ['pyorderedfuzzy/ofmodels/*.so']},
    url='',
    license='',
    author='amarszalek',
    author_email='amarszalek@pk.edu.pl',
    description='Python package for Ordered Fuzzy Numbers'
)
