from subprocess import call

call('swig -python -py3 pyorderedfuzzy/ofmodels/src/objective.i', shell=True)
call('python pyorderedfuzzy/ofmodels/src/setup.py build_ext build-lib=pyorderedfuzzy/ofmodels/src/ build-temp=pyorderedfuzzy/ofmodels/src/', shell=True)
