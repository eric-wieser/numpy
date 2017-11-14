#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"
#include "numpyos.h"

/* This is a backport of Py_SETREF */
#define NPY_SETREF(op, op2)                      \
    do {                                        \
        PyObject *_py_tmp = (PyObject *)(op);   \
        (op) = (op2);                           \
        Py_DECREF(_py_tmp);                     \
    } while (0)


/*
 * Heavily derived from PyLong_FromDouble
 * Notably, we can't set the digits directly, so have to shift and or instead.
 */
PyObject *
npy_longdouble_to_PyLong(npy_longdouble ldval)
{
    PyObject *v;
    PyObject *l_chunk_size;
    /*
     * number of bits to extract at a time. CPython uses 30, but that's because
     * it's tied to the internal long representation
     */
    const int chunk_size = NPY_BITSOF_LONGLONG;
    npy_longdouble frac;
    int i, ndig, expo, neg;
    neg = 0;

    if (npy_isinf(ldval)) {
        PyErr_SetString(PyExc_OverflowError,
                        "cannot convert longdouble infinity to integer");
        return NULL;
    }
    if (npy_isnan(ldval)) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot convert longdouble NaN to integer");
        return NULL;
    }
    if (ldval < 0.0) {
        neg = 1;
        ldval = -ldval;
    }
    frac = npy_frexpl(ldval, &expo); /* ldval = frac*2**expo; 0.0 <= frac < 1.0 */
    v = PyLong_FromLong(0L);
    if (v == NULL)
        return NULL;
    if (expo <= 0)
        return v;

    ndig = (expo-1) / chunk_size + 1;

    l_chunk_size = PyLong_FromLong(chunk_size);
    if (l_chunk_size == NULL) {
        Py_DECREF(v);
        return NULL;
    }

    /* Get the MSBs of the integral part of the float */
    frac = npy_ldexpl(frac, (expo-1) % chunk_size + 1);
    for (i = ndig; --i >= 0; ) {
        npy_ulonglong chunk = (npy_ulonglong)frac;
        PyObject *l_chunk;
        /* v = v << chunk_size */
        NPY_SETREF(v, PyNumber_Lshift(v, l_chunk_size));
        if (v == NULL) {
            goto done;
        }
        l_chunk = PyLong_FromUnsignedLongLong(chunk);
        if (l_chunk == NULL) {
            Py_DECREF(v);
            v = NULL;
            goto done;
        }
        /* v = v | chunk */
        NPY_SETREF(v, PyNumber_Or(v, l_chunk));
        Py_DECREF(l_chunk);
        if (v == NULL) {
            goto done;
        }

        /* Remove the msbs, and repeat */
        frac = frac - (npy_longdouble) chunk;
        frac = npy_ldexpl(frac, chunk_size);
    }

    /* v = -v */
    if (neg) {
        NPY_SETREF(v, PyNumber_Negative(v));
        if (v == NULL) {
            goto done;
        }
    }

done:
    Py_DECREF(l_chunk_size);
    return v;
}

/* Helper function to get unicode(PyLong).encode('utf8') */
static PyObject *
_PyLong_Bytes(PyObject *long_obj) {
    PyObject *bytes;
#if defined(NPY_PY3K)
    PyObject *unicode = PyObject_Str(long_obj);
    if (unicode == NULL) {
        return NULL;
    }
    bytes = PyUnicode_AsUTF8String(unicode);
    Py_DECREF(unicode);
#else
    bytes = PyObject_Str(long_obj);
#endif
    return bytes;
}


/**
 * TODO: currently a hack that converts the long through a string. This is
 * correct, bus slow.
 *
 * To do this right, we need to know the number of digits in the mantissa, so
 * as to compute rounding modes correctly. PyLong_AsDouble shows how some of
 * this can work.
 */
npy_longdouble
npy_longdouble_from_PyLong(PyObject *long_obj) {
    npy_longdouble result;
    char *end;
    char *s;
    PyObject *bytes;

    bytes = _PyLong_Bytes(long_obj);
    if (bytes == NULL){
        return -1;
    }

    s = PyBytes_AsString(bytes);
    end = NULL;
    errno = 0;
    result = NumPyOS_ascii_strtold(cstr, &end);
    if (errno == ERANGE) {
        /* strtold returns INFINITY of the correct sign. */
        if (PyErr_Warn(PyExc_RuntimeWarning,
                "overflow encountered in conversion from string") < 0) {
            result = -1;
        }
    }
    else if (errno) {
        PyErr_Format(PyExc_RuntimeError,
                     "Could not parse long as longdouble: %s (%s)",
                     s,
                     strerror(errno));
        result = -1;
    }

    /* Extra characters at the end of the string, or nothing parsed */
    if (end == s || *end) {
        PyErr_Format(PyExc_RuntimeError,
                     "Could not parse long as longdouble: %s",
                     s);
        result = -1;
    }
    Py_DECREF(bytes);

    return result;
}
