from subprocess import call

call('swig -python -py3 pyorderedfuzzy/ofmodels/src/objective.i', shell=True)
call('python pyorderedfuzzy/ofmodels/src/setup.py build_ext --inplace', shell=True)
