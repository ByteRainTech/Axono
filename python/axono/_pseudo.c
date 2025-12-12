/* _pseudo.c : dummy extension module for setuptools platform tagging */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyModuleDef pseudomodule = {
    PyModuleDef_HEAD_INIT,
    "_pseudo",              /* m_name  */
    "dummy module",         /* m_doc   */
    -1,                     /* m_size  */
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__pseudo(void)
{
    return PyModule_Create(&pseudomodule);
}
