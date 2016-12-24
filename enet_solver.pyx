# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Alexis Mignon <alexis.mignon@gmail.com>
#         Manoj Kumar <manojkumarsivaraj334@gmail.com>
#
# Licence: BSD 3 clause

from libc.math cimport fabs, sqrt
cimport numpy as np
import numpy as np
import numpy.linalg as linalg

cimport cython
from cpython cimport bool
import warnings

ctypedef np.float64_t DOUBLE
ctypedef np.uint32_t UINT32_t
ctypedef np.uint8_t UINT8_t

np.import_array()

DEF USE_MKL = 1


cdef inline double fmax(double x, double y) nogil:
    if x > y:
        return x
    return y


cdef inline double fsign(double f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


cdef double abs_max(int n, double* a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef double m = fabs(a[0])
    cdef double d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef double max(int n, double* a) nogil:
    """np.max(a)"""
    cdef int i
    cdef double m = a[0]
    cdef double d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


cdef double diff_abs_max(int n, double* a, double* b) nogil:
    """np.max(np.abs(a - b))"""
    cdef int i
    cdef double m = fabs(a[0] - b[0])
    cdef double d
    for i in range(1, n):
        d = fabs(a[i] - b[i])
        if d > m:
            m = d
    return m


IF USE_MKL==1:
    cdef extern from "cblas.h":
        enum CBLAS_ORDER:
            CblasRowMajor=101
            CblasColMajor=102
        enum CBLAS_TRANSPOSE:
            CblasNoTrans=111
            CblasTrans=112
            CblasConjTrans=113
            AtlasConj=114

        void daxpy "cblas_daxpy"(int N, double alpha, double *X, int incX,
                                 double *Y, int incY) nogil
        double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY
                                 ) nogil
        double dasum "cblas_dasum"(int N, double *X, int incX) nogil

ELSE:
    cdef void daxpy(int n, double alpha, double* X, int incX, double* Y, int incY) nogil:
        """ y := a*x + y """
        cdef int i
        cdef int x_i = 0
        cdef int y_i = 0
        for i in range(n):
            Y[i] = alpha*X[i] + Y[i]
            x_i += incX
            y_i += incY

    cdef double dasum (int n, double* X, int incX) nogil:
        """ res = |Re x(1)| + |Im x(1)| + |Re  x(2)| + |Im  x(2)|+ ... + |Re  x(n)| + |Im x(n)|, """
        cdef int i
        cdef double m = fsign(X[0])*X[0]
        cdef int x_i = incX
        for i in range(1, n):
            if X[x_i]<0:
                m += -X[i]
            else:
                m += +X[i]
            x_i += incX
        return m

    cdef double ddot(int n, double* X, int incX, double* Y, int incY) nogil:
        """ res = sum_i x_i y_i """
        cdef int i
        cdef double m = X[0]*Y[0]
        cdef int x_i = incX
        cdef int y_i = incY
        for i in range(1, n):
            m += X[x_i]*Y[y_i]
            x_i += incX
            y_i += incY
        return m



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def enet_coordinate_descent(np.ndarray[DOUBLE, ndim=1] w,
                            double alpha, double beta,
                            np.ndarray[DOUBLE, ndim=2] X,
                            np.ndarray[DOUBLE, ndim=1] y,
                            int max_iter, double tol, int use_active_set):
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression

        We minimize

        1 norm(y - X w, 2)^2 + alpha norm(w, 1) + beta norm(w, 2)^2
        -                                         ----
        2                                           2

    """

    # get the data information into easy vars
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]

    # get the number of tasks indirectly, using strides
    cdef unsigned int n_tasks = y.strides[0] / sizeof(DOUBLE)

    # compute norms of the columns of X
    cdef np.ndarray[DOUBLE, ndim=1] norm_cols_X = (X**2).sum(axis=0)

    # initial value of the residuals
    cdef np.ndarray[DOUBLE, ndim=1] R = np.empty(n_samples)

    cdef np.ndarray[DOUBLE, ndim=1] XtA = np.empty(n_features)
    cdef double tmp
    cdef double w_ii
    cdef double d_w_max
    cdef double w_max
    cdef double d_w_ii
    cdef double gap = tol + 1.0
    cdef double d_w_tol = tol
    cdef double dual_norm_XtA
    cdef double R_norm2
    cdef double w_norm2
    cdef double l1_norm
    cdef unsigned int ii
    cdef unsigned int i
    cdef unsigned int n_iter
    cdef unsigned int f_iter

    cdef unsigned int total_evals = 0

    cdef unsigned int active_set_reevals = 0
    cdef unsigned int n_active_set_changes = 0
    cdef bool build_active_set = True
    
    cdef np.ndarray[UINT8_t, ndim=1] active_set = np.zeros(n_features, dtype='uint8')

    if use_active_set==0:
        active_set = np.ones(n_features, dtype='uint8')

    if alpha == 0:
        warnings.warn("Coordinate descent with alpha=0 may lead to unexpected"
            " results and is discouraged.")

    #with gil:

    # R = y - np.dot(X, w)
    for i in range(n_samples):
        R[i] = y[i] - ddot(n_features, <DOUBLE*>(X.data + i * sizeof(DOUBLE)), n_samples, <DOUBLE*>w.data, 1)

    # tol *= np.dot(y, y)
    tol *= ddot(n_samples, <DOUBLE*>y.data, n_tasks,
                <DOUBLE*>y.data, n_tasks)

    gapcount = 0
    for n_iter in range(max_iter):
        count = 0

        w_max = 0.0
        d_w_max = 0.0


        for f_iter in range(n_features):  # Loop over coordinates
            
            # check if feature belongs to the active set
            if build_active_set==False and active_set[f_iter]==0:
                continue
            # statistics
            total_evals += 1

            ii = f_iter


            if norm_cols_X[ii] == 0.0:
                continue

            w_ii = w[ii]  # Store previous value

            if w_ii != 0.0:
                # R += w_ii * X[:,ii]
                daxpy(n_samples, w_ii, <DOUBLE*>(X.data + ii * n_samples * sizeof(DOUBLE)), 1, <DOUBLE*>R.data, 1)

            # tmp = (X[:,ii]*R).sum()
            tmp = ddot(n_samples, <DOUBLE*>(X.data + ii * n_samples * sizeof(DOUBLE)), 1, <DOUBLE*>R.data, 1)
            w[ii] = (fsign(tmp) * fmax(fabs(tmp) - alpha, 0) / (norm_cols_X[ii] + beta))

            # first iteration: select active set
            if build_active_set==True and use_active_set>0:
                if fabs(w[ii]-w_ii)>1e-10:
                    if active_set[ii]==0: 
                        n_active_set_changes += 1
                    active_set[ii] = 1

            if w[ii] != 0.0:
                # R -=  w[ii] * X[:,ii] # Update residual
                daxpy(n_samples, -w[ii], <DOUBLE*>(X.data + ii * n_samples * sizeof(DOUBLE)), 1, <DOUBLE*>R.data, 1)

            # update the maximum absolute coefficient update
            d_w_ii = fabs(w[ii] - w_ii)
            if d_w_ii > d_w_max:
                d_w_max = d_w_ii

            if fabs(w[ii]) > w_max:
                w_max = fabs(w[ii])

        # next: evaluate the active set
        build_active_set = False

#        if (n_iter % 100)==0:
#            # R = y - np.dot(X, w)
#            for i in range(n_samples):
#                R[i] = y[i] - ddot(n_features, <DOUBLE*>(X.data + i * sizeof(DOUBLE)), n_samples, <DOUBLE*>w.data, 1)


        if (w_max==0.0 or d_w_max/w_max<d_w_tol or n_iter==max_iter-1):
            # the biggest coordinate update of this iteration was smaller
            # than the tolerance: check the duality gap as ultimate
            # stopping criterion   

            # reactivate active set 
            if active_set_reevals<5 or n_active_set_changes>0:
                build_active_set = True
                n_active_set_changes = 0
                active_set_reevals += 1
            else:
                if (n_iter % 50)==0:
                    build_active_set = True
                    n_active_set_changes = 0
                    active_set_reevals += 1


            # XtA = np.dot(X.T, R) - beta * w
            for i in range(n_features):
                XtA[i] = ddot(n_samples, <DOUBLE*>(X.data + i * n_samples *sizeof(DOUBLE)), 1, <DOUBLE*>R.data, 1) - beta * w[i]

            dual_norm_XtA = abs_max(n_features, <DOUBLE*>XtA.data)

            # R_norm2 = np.dot(R, R)
            R_norm2 = ddot(n_samples, <DOUBLE*>R.data, 1, <DOUBLE*>R.data, 1)

            # w_norm2 = np.dot(w, w)
            w_norm2 = ddot(n_features, <DOUBLE*>w.data, 1, <DOUBLE*>w.data, 1)

            if (dual_norm_XtA > alpha):
                const = alpha / dual_norm_XtA
                A_norm2 = R_norm2 * (const ** 2)
                gap = 0.5 * (R_norm2 + A_norm2)
            else:
                const = 1.0
                gap = R_norm2

            l1_norm = dasum(n_features, <DOUBLE*>w.data, 1)

            # np.dot(R.T, y)
            gap += (alpha * l1_norm - const * ddot(
                        n_samples, <DOUBLE*>R.data, 1, <DOUBLE*>y.data, n_tasks) + 0.5 * beta * (1 + const ** 2) * (w_norm2))

            if gap < tol:
                # return if we reached desired tolerance
                break

    return w, gap, tol, n_iter+1, total_evals