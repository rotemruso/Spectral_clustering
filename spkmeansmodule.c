#define PY_SSIZE_T_CLEAN /* For all # variants of unit formats (s#, y#, etc.) use Py_ssize_t rather than int. */
#include <Python.h>      /* MUST include <Python.h>, this implies inclusion of the following standard headers:
                             <stdio.h>, <string.h>, <errno.h>, <limits.h>, <assert.h> and <stdlib.h> (if available). */
                         /* include <Python.h> has to be before any standard headers are included */
#include "spkmeans.h"

/* A function that converts a PyObject matrix of floats and returns a 2d array of double.
   The function assumes that the PyObject is indeed a list of lists */
double **convert_PyObject_matrix_to_list(PyObject *matrix, int rows, int columns)
{
    double **c_matrix = two_dimensional_alloc(rows, columns);
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns; j++)
        {
            c_matrix[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(matrix, i), j));
        }
    }

    return c_matrix;
}

/* 
    A function that converts a 2d double array to a list of list of floats 
 */
PyObject *convert_matrix_to_PyObject(double **matrix, int rows, int columns)
{
    PyObject *py_matrix = PyList_New(rows);
    int i, j;

    for (i = 0; i < rows; i++)
    {
        PyObject *row = PyList_New(columns);
        for (j = 0; j < columns; j++)
        {
            PyList_SetItem(row, j, PyFloat_FromDouble(matrix[i][j]));
        }
        PyList_SetItem(py_matrix, i, row);
    }
    return py_matrix;
}

/*
 * This actually defines the  calculate_eigenvalues function using a wrapper C API function
 * The wrapping function needs a PyObject* self argument.
 * This is a requirement for all functions and methods in the C API.
 * It has input PyObject *args from Python.
 */
static PyObject *calculate_eigenvalues_capi(PyObject *self, PyObject *args)
{
    PyObject *matrix;
    PyObject *py_eigenvectors, *py_eigenvalues;
    double **c_matrix;
    double **eigenvectors;
    int size;
    int i;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &matrix))
    {
        printf("Invalid Input!\n");
        return NULL;
    }
    if (!PyList_Check(matrix))
    {
        printf("Invalid Input!\n");
        return NULL;
    }
    if (!PyList_Check(PyList_GetItem(matrix, 0)))
    {
        printf("Invalid Input!\n");
        return NULL;
    }

    size = (int)PyList_Size(matrix);
    c_matrix = convert_PyObject_matrix_to_list(matrix, size, size);
    eigenvectors = calculate_eigenvalues(c_matrix, size);
    py_eigenvectors = convert_matrix_to_PyObject(eigenvectors, size, size);
    py_eigenvalues = PyList_New(size);
    for (i = 0; i < size; i++)
    {
        PyList_SetItem(py_eigenvalues, i, PyFloat_FromDouble(c_matrix[i][i]));
    }

    free_matrix(c_matrix);
    free_matrix(eigenvectors);
    return Py_BuildValue("(OO)", py_eigenvalues, py_eigenvectors);
}

static PyObject *kmeans_clustering_capi(PyObject *self, PyObject *args)
{
    PyObject *points, *centers;
    PyObject *new_centers;
    double **c_new_centers;
    double **c_points, **c_centers;
    int k, num_coordinates, num_points;

    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &points, &PyList_Type, &centers))
    {
        printf("Invalid Input!\n");
        return NULL;
    }
    if (!PyList_Check(points))
    {
        printf("Invlid Input!\n");
        return NULL;
    }
    if (!PyList_Check(PyList_GetItem(points, 0)))
    {
        printf("Invlid Input!\n");
        return NULL;
    }
    if (!PyList_Check(centers))
    {
        printf("Invlid Input!\n");
        return NULL;
    }
    if (!PyList_Check(PyList_GetItem(centers, 0)))
    {
        printf("Invlid Input!\n");
        return NULL;
    }

    k = (int)PyList_Size(centers);
    num_points = (int)PyList_Size(points);
    num_coordinates = (int)PyList_Size(PyList_GetItem(points, 0));
    c_points = convert_PyObject_matrix_to_list(points, num_points, num_coordinates);
    c_centers = convert_PyObject_matrix_to_list(centers, k, num_coordinates);
    c_new_centers = kmeans_clustering(c_points, c_centers, k, num_points, num_coordinates);
    if (!c_new_centers)
    {
        return NULL;
    }
    new_centers = convert_matrix_to_PyObject(c_new_centers, k, num_coordinates);
    free_matrix(c_new_centers);
    free_matrix(c_centers);
    free_matrix(c_points);
    return new_centers;
}

static PyObject *calculate_W_capi(PyObject *self, PyObject *args)
{
    double **c_data;
    int num_points;
    int num_coordinates;
    double **c_W;
    PyObject *data, *W;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &data))
    {
        printf("Invalid Input!\n");
        return NULL;
    }
    if (!PyList_Check(data))
    {
        printf("Invlid Input!\n");
        return NULL;
    }
    if (!PyList_Check(PyList_GetItem(data, 0)))
    {
        printf("Invlid Input!\n");
        return NULL;
    }
    num_points = (int)PyList_Size(data);
    num_coordinates = (int)PyList_Size(PyList_GetItem(data, 0));
    c_data = convert_PyObject_matrix_to_list(data, num_points, num_coordinates);
    c_W = calculate_weighted_adjency_matrix(c_data, num_points, num_coordinates);
    free_matrix(c_data);
    W = convert_matrix_to_PyObject(c_W, num_points, num_points);
    free_matrix(c_W);
    return W;
}

static PyObject *calculate_D_capi(PyObject *self, PyObject *args)
{
    double **c_W;
    double **c_D;
    PyObject *D, *W;
    int size;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &W))
    {
        printf("Invalid Input!\n");
        return NULL;
    }
    if (!PyList_Check(W))
    {
        printf("Invlid Input!\n");
        return NULL;
    }
    if (!PyList_Check(PyList_GetItem(W, 0)))
    {
        printf("Invlid Input!\n");
        return NULL;
    }

    size = (int)PyList_Size(W);
    c_W = convert_PyObject_matrix_to_list(W, size, size);
    c_D = calculate_diagonal_degree_matrix(c_W, size);

    D = convert_matrix_to_PyObject(c_D, size, size);
    free_matrix(c_W);
    free_matrix(c_D);
    return D;
}

static PyObject *calculate_lnorm_capi(PyObject *self, PyObject *args)
{
    PyObject *W, *D_minus_half;
    PyObject *Lnorm;
    double **c_W, **c_D_minus_half, **c_Lnorm;
    int size;
    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &W, &PyList_Type, &D_minus_half))
    {
        printf("Invlid Input!\n");
        return NULL;
    }
    if (!PyList_Check(W))
    {
        printf("Invlid Input!\n");
        return NULL;
    }
    if (!PyList_Check(PyList_GetItem(W, 0)))
    {
        printf("Invlid Input!\n");
        return NULL;
    }
    if (!PyList_Check(D_minus_half))
    {
        printf("Invlid Input!\n");
        return NULL;
    }
    if (!PyList_Check(PyList_GetItem(D_minus_half, 0)))
    {
        printf("Invlid Input!\n");
        return NULL;
    }

    size = (int)PyList_Size(W);
    c_W = convert_PyObject_matrix_to_list(W, size, size);
    c_D_minus_half = convert_PyObject_matrix_to_list(D_minus_half, size, size);
    c_Lnorm = calculate_lnorm(c_W, c_D_minus_half, size);
    free_matrix(c_W);
    free_matrix(c_D_minus_half);
    Lnorm = convert_matrix_to_PyObject(c_Lnorm, size, size);
    free_matrix(c_Lnorm);
    return Lnorm;
}

/*
 * This array tells Python what methods this module has.
 * We will use it in the next structure
 */
static PyMethodDef capiMethods[] = {
    {"calculate_eigenvalues",                 /* the Python method name that will be used */
     (PyCFunction)calculate_eigenvalues_capi, /* the C-function that implements the Python function and returns static PyObject*  */

     METH_VARARGS, /* flags indicating parametersaccepted for this function */
     PyDoc_STR("calculate_eigenvalues")},
    {
        "kmeans",                            // name exposed to Python
        (PyCFunction)kmeans_clustering_capi, // C wrapper function
        METH_VARARGS,                        // received variable args (but really just 1)
        "returns the final centers"          // documentation
    },                                       /*  The docstring for the function */
    {
        "calculate_W",
        (PyCFunction)calculate_W_capi,
        METH_VARARGS,
        "calculates the weighted adjacency matrix"},
    {"calculate_D",
     (PyCFunction)calculate_D_capi,
     METH_VARARGS,
     "calculates the diagonal degree matrix"},
    {"calculate_Lnorm",
     (PyCFunction)calculate_lnorm_capi,
     METH_VARARGS,
     "calculates the Lnorm matrix"},
    {NULL, NULL, 0, NULL} /* The last entry must be all NULL as shown to act as a
                                 sentinel. Python looks for this entry to know that all
                                 of the functions for the module have been defined. */
};

/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spkmeansmodule", /* name of module */
    NULL,             /* module documentation, may be NULL */
    -1,               /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    capiMethods       /* the PyMethodDef array from before containing the methods of the extension */
};

/*
 * The PyModuleDef structure, in turn, must be passed to the interpreter in the module’s initialization function.
 * The initialization function must be named PyInit_name(), where name is the name of the module and should match
 * what we wrote in struct PyModuleDef.
 * This should be the only non-static item defined in the module file
 */
PyMODINIT_FUNC PyInit_spkmeansmodule(void)
{
    PyObject *m;

    m = PyModule_Create(&moduledef);
    if (!m)
    {
        return NULL;
    }
    return m;
}
