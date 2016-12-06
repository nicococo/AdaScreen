import scipy.sparse as sparse
import sklearn.linear_model as lm
import numpy as np
import time

from screening_rules import ScreenDummy
from solver import SklearnCDSolver


class ScreeningElasticNetPath(object):
    """ Elastic net solver with lambda path and screening. """

    # screening options/parameters
    alg_screen = None

    one_shot_screening = False  # Instead of using the actual last lambda parameter as l0, we set 
                                # l0 always to lmax (one-shot screening). Setting this to true will decrease screening performance,
                                # since the distance between the actual lambda and lmax will increase in 
                                # each iteration.

    use_warm_start = True # This should be always true, otherwise training time will increase tremendously.

    # lambda path options
    path = None
    path_lb = 0.7
    path_ub = 1.0
    path_steps = 10
    path_stepsize = 0.9
    path_scale = 'geometric'

    # the actual solver
    solver = None


    def __init__(self, alg_screen, solver, \
        path_lb=0.7, path_ub=1.0, path_steps=10, path_stepsize=0.9, path_scale='geometric'):
        # screening options 
        if alg_screen==None:
            self.alg_screen = ScreenDummy()
        else:
            self.alg_screen = alg_screen
        # lambda path options
        self.path_lb = path_lb
        self.path_ub = path_ub
        self.path_steps = path_steps 
        self.path_stepsize = path_stepsize
        self.path_scale = path_scale 
        # if not specified, use the active set cd solver        
        if solver == None:
            self.solver = SklearnCDSolver()
        else:
            self.solver = solver

    def set_solver(self, solver):
        self.solver = solver

    def get_plain_path(self, use_geom_hard_lower_bound=False):
        if self.path_scale=='linear':
            path = np.linspace(self.path_ub, self.path_lb, self.path_steps)
        if self.path_scale=='log':
            path = np.logspace(np.log10(self.path_ub), np.log10(self.path_lb), self.path_steps)
        if self.path_scale=='geometric':
            print np.linspace(self.path_ub, self.path_lb, self.path_steps)
            path = np.linspace(self.path_ub, self.path_lb, self.path_steps)
            for i in range(1, self.path_steps):
                path[i] = path[i-1]*self.path_stepsize
                if path[i]<self.path_lb:
                    if use_geom_hard_lower_bound:
                        path[i] = self.path_lb
                    path = path[:i]
                    break
            print path
        return path

    def calc_lambda_max(self, X, y):
        vals = np.abs(X.dot(y))
        lmax_ind = int(np.argmax(vals))
        lmax = float(vals[lmax_ind])
        lmax_x = X[lmax_ind,:]
        
        #print lmax_x.dot(y)
        #print -lmax_x.dot(y)
        #print vals[lmax_ind]

        print lmax_x.dot(y)
        if lmax_x.dot(y)<0.0:
            print -lmax_x.dot(y)
            lmax_x = -lmax_x
        return (lmax, lmax_ind, lmax_x)
    
    def fit(self, X, y, l2=0.0, max_iter=1000, tol=1e-6, debug=True):
        if sparse.issparse(X):
            raise Exception('SPARSE MATRIX NOT SUPPORTED YET')
 
        # init
        (lmax, lmax_ind, lmax_x) = self.calc_lambda_max(X,y)
        path = self.get_plain_path() * lmax

        # screening args
        normy = np.linalg.norm(y)
        normX = np.linalg.norm(X, axis=1)

        P = len(path)
        (DIMS, EXMS) = X.shape
        print DIMS
        print EXMS
        res = np.zeros((DIMS, P)) # intermediate solutions (beta for all lambda) 
        scr_inds = [] # indices of non-discarded entries after screening
        nz_inds = [] # indices of non-zero entries after training
        intervals = [] # screening intervals used by IADPP

        # screening init
        self.alg_screen.init(lmax, lmax_x, X, y, normX, normy, path)

        # solver init
        self.solver.init(lmax, lmax_x, X, y, path)

        # solution for lambda_max
        scr_inds.append([lmax_ind])
        nz_inds.append([])
        intervals.append(np.ones(DIMS)*lmax)
        
        # some additional information
        times_solver = []
        times_screening = []

        if debug:
            print('Debug is turned on.')

        for i in range(1,P):
            # (a) remove variables either one-shot screening or sequential
            startTime = time.time()
            if self.one_shot_screening:
                (inds, interval) = self.alg_screen.screen(path[i], lmax, lmax, lmax_x, np.zeros(DIMS), X, y, normX, normy, nz_inds[0], intervals[0])
            else:
                (inds, interval) = self.alg_screen.screen(path[i], path[i-1], lmax, lmax_x, res[:,i-1], X, y, normX, normy, nz_inds[i-1], intervals[i-1])
            times_screening.append(time.time()-startTime)
            intervals.append(interval)

            # (b) solve lasso for current lambda using hot-start 'beta'
            startTime = time.time()
            if self.use_warm_start:
                (res[inds,i], iters, gap) = self.solver.solve(res[inds,i-1], X[inds,:].T, y, path[i], l2, max_iter=max_iter, tol=tol)
            else:
                (res[inds,i], iters, gap) = self.solver.solve(res[inds,0], X[inds,:].T, y, path[i], l2, max_iter=max_iter, tol=tol)
            times_solver.append(time.time()-startTime)

            # (c) re-construct parameter vector
            scr_inds.append(inds)
            nz_inds.append(np.where(np.abs(res[:, i]) >= 1e-30)[0])
            print('Iter{0}: {1} non-zeros after lasso, {2} after screening.'.format(i, len(nz_inds[-1]), len(inds)))
            # (optional) (d) in debug mode, check if screening does not discard non-zero betas
            if debug:
                solver = lm.Lasso(alpha=path[i]/float(EXMS), normalize=False, fit_intercept=False, tol=tol, max_iter=50000)
                solver.fit(X.T, y)
                set_true = set(np.where(abs(solver.coef_)>=tol)[0])
                # test for screening error
                set_scr = set(inds)
                if not set_true.issubset(set_scr):
                    print set_true
                    print set_scr
                    finds = set_true.difference(set_scr)
                    print finds
                    print solver.coef_[int(finds.pop())]
                    raise AssertionError('Screen discarded non-zero coefficients!')
                # test for solver error
                delta = np.linalg.norm(solver.coef_-res[:,i])
                if delta > 2.0*tol*DIMS:
                    print delta
                    print delta/DIMS
                    print ('Warning !Solver solution differs!')
        # delete screening caches, if any
        self.alg_screen.release()
        self.solver.release()
        return (res, nz_inds, scr_inds, path, times_solver, times_screening)

    def predict(self, X):
        raise NotImplementedError

