import numpy as np
import scipy.sparse as sparse
from sets import Set

from screening_lasso_path import ScreeningLassoPath

class NeighborSelect(ScreeningLassoPath):

    def __init__(self, alg_screen, solver, path_lb=0.7, path_ub=1.0, path_steps=10, path_stepsize=0.9, path_scale='geometric'):
        ScreeningLassoPath.__init__(self, alg_screen, solver, path_lb=path_lb, path_ub=path_ub, path_steps=path_steps, path_stepsize=path_stepsize, path_scale=path_scale)


    def fit(self, X):
        (EXMS, DIMS) = X.shape

        # non-zero entries for all lambdas along the path [path x sparse(dims x dims)]
        Cb = []
        for p in range(self.path_steps):
            Cb.append(sparse.lil_matrix((DIMS, DIMS), dtype='i'))

        for i in range(DIMS):
            inds = range(DIMS)
            inds.remove(i) 

            myLasso = ScreeningLassoPath(self.alg_screen, self.solver, path_lb=self.path_lb, \
                path_ub=self.path_ub, path_steps=self.path_steps, path_stepsize=self.path_stepsize, path_scale=self.path_scale)
            (res, nz_inds, foo1, foo2, t1, t2) = myLasso.fit(X[:,inds].T, X[:,i], debug=False) 
            
            # first entry in nz_inds is empty (lambda_max has no non-zero solution)
            for p in range(1,len(nz_inds)):
                con = Cb[p]
                inds = np.where(nz_inds[p]>=i)[0].astype(int)
                nz_inds[p][inds] += 1
                # OR connection: [:,:]>=1 -> add edge
                # AND connection: [:,:]==2 -> add edge
                con[i,nz_inds[p]] = con[i,nz_inds[p]].todense() + 1
                con[nz_inds[p],i] = con[nz_inds[p],i].todense() + 1
        return Cb

