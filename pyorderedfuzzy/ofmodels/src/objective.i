//objective.i

%module cstuff

%{
	#define SWIG_FILE_WITH_INIT
	#include "objective.h"
%}
%include "numpy.i"

%init %{
  import_array();
%}

//wektor p
%apply (double* IN_ARRAY1, int DIM1) {(double* p, int np)}
//wektor xx
%apply (double* IN_ARRAY1, int DIM1) {(double* xx, int nx)}
//wektor yy
%apply (double* IN_ARRAY1, int DIM1) {(double* yy, int ny)}
//wektor wyjsciowy
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* grad, int ng)}


%include "objective.h"