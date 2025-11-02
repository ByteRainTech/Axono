#include <Python.h>
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_pseudo", NULL, -1, NULL};
PyMODINIT_FUNC PyInit__pseudo(void){ return PyModule_Create(&module); }
