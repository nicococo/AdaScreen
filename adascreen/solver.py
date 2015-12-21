import numpy as np
import sklearn.linear_model as lm

# un-comment if you have compile enet_solver
import enet_solver

# un-comment if you have installed glmnet
#import glmnet.glmnet as glmnet


class AbstractSolver(object):
    """ Wrapper to enable the use of various solver for ElasticNet and Lasso. """
    
    name = 'AbstractSolver'

    def __init__(self, name):
        self.name = name

    def init(self, lmax, lmax_x, X, y, path):
        print('{0}: nothing to initialize.'.format(self.name))

    def solve(self, start_pos, X, y, l1_reg, l2_reg, max_iter=20000, tol=1e-6):
        # return (new_coefs, iterations, dual_gap)  
        # set to zero, if not applicable
        pass

    def release(self):
        print('{0}: nothing to release.'.format(self.name))

    def __str__(self):
        return '{0}'.format(self.name)



class SklearnCDSolver(AbstractSolver):
    """ Wrapper for sklearn solver for ElasticNet and Lasso. """

    def __init__(self):
        AbstractSolver.__init__(self, 'SklearnCDSolver')

    def solve(self, start_pos, X, y, l1_reg, l2_reg, max_iter=20000, tol=1e-6):
        (coefs, dual_gap, eps) = lm.cd_fast.enet_coordinate_descent(start_pos, l1_reg, l2_reg, np.asfortranarray(X), y, max_iter, tol, False)
        return (coefs, 0, dual_gap)


class SklearnLarsSolver(AbstractSolver):
    """ Wrapper for sklearn solver for ElasticNet and Lasso. """

    def __init__(self):
        AbstractSolver.__init__(self, 'SklearnLarsSolver')

    def solve(self, start_pos, X, y, l1_reg, l2_reg, max_iter=20000, tol=1e-6):
        lars = lm.LassoLars(alpha=l1_reg / float(y.size), max_iter=max_iter, fit_path=False, normalize=False, fit_intercept=False)
        lars.coef_ = start_pos
        lars.fit(X, y)
        return (lars.coef_[0], 0, 0)


class ActiveSetCDSolver(AbstractSolver):
    """ Wrapper for our refined sklearn solver for ElasticNet and Lasso. """
    use_active_set = 1

    def __init__(self, use_active_set=1):
        postfix = ''
        if use_active_set==0:
            postfix = ' (active set disabled)'
        AbstractSolver.__init__(self, 'ActiveSetCDSolver{0}'.format(postfix))
        self.use_active_set = use_active_set

    def solve(self, start_pos, X, y, l1_reg, l2_reg, max_iter=20000, tol=1e-6):
        (coefs, gap, eps, iters, n_evals) = enet_solver.enet_coordinate_descent(start_pos, l1_reg, l2_reg, np.asfortranarray(X), y, max_iter, tol, self.use_active_set)
        print iters
        return (coefs, iters, gap)


class GlmnetSolver(AbstractSolver):
    """ Wrapper for Fortran GLMNET Lasso. """
    lmax = 1.0

    def __init__(self):
        AbstractSolver.__init__(self, 'Glmnet')

    def init(self, lmax, lmax_x, X, y, path):
        self.lmax = lmax
        print('{0}: initialization.'.format(self.name))

    def solve(self, start_pos, X, y, l1_reg, l2_reg, max_iter=20000, tol=1e-6):
        n_lambdas, intercept_, coef_, _, _, rsquared_, lambdas, _, jerr  = \
            glmnet.elastic_net(np.asfortranarray(X), np.asfortranarray(y), \
            1.0, flmin=l1_reg/self.lmax, standardize=False)
        return (coef_[:,0], 0, 0.0)
