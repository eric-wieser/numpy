#ifndef __NPY_LONGDOUBLE_H
#define __NPY_LONGDOUBLE_H

#include "npy_config.h"
#include "numpy/ndarraytypes.h"

/* Convert a npy_longdouble to a python `long` integer.
 *
 * Results are rounded towards zero.
 *
 * This performs the same task as PyLong_FromDouble, but for long doubles
 * which have a greater range.
 */
NPY_NO_EXPORT PyObject *
npy_longdouble_to_PyLong(npy_longdouble ldval);

/* Convert a python `long` integer to a npy_longdouble
 *
 * This performs the same task as PyLong_AsDouble, but for long doubles
 * which have a greater range.
 *
 * Returns -1 if an error occurs.
 */
NPY_NO_EXPORT npy_longdouble
npy_longdouble_from_PyLong(PyObject *long_obj);

#endif
