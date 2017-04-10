from subprocess import call

call("python pyorderedfuzzy/ofmodels/src/setup.py build_ext --build-lib='pyorderedfuzzy/ofmodels' --build-temp='pyorderedfuzzy/ofmodels/src'", shell=True)
call("python setup.py install", shell=True)
