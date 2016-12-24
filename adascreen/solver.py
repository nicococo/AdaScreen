import numpy as np
import sklearn.linear_model as lm

# un-comment if you have compile enet_solver
import imp
try:
    import pyximport; pyximport.install()
    imp.find_module('enet_solver')
    found_enet_solver = True
    import enet_solver
except ImportError:
    found_enet_solver = False

# un-comment if you have installed glmnet
import imp
try:
    imp.find_module('glmnet')
    found_glmnet_solver = True
    import glmnet.glmnet as glmnet
except ImportError:
    found_glmnet_solver = False


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
        # (coefs, dual_gap, eps) = lm.cd_fast.enet_coordinate_descent(start_pos, l1_reg, l2_reg, np.asfortranarray(X), y, max_iter, tol, False)
        # coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random, positive)
        coefs, dual_gap, eps, n_iter_ = lm.cd_fast.enet_coordinate_descent(
            start_pos, l1_reg, l2_reg, np.asfortranarray(X), y, max_iter, tol, np.random, 0, 0)
        return coefs, 0, dual_gap


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


class ProximalGradientSolver(AbstractSolver):
    """ Proximal gradient solver for elastic net. """
    def __init__(self):
        AbstractSolver.__init__(self, 'ProxGradientSolver')

    def solve(self, start_pos, X, y, l1_reg, l2_reg, max_iter=20000, tol=1e-9, debug=False):
        # X \in R^(samples x feats)
        # y \in R^samples
        max_iter = 20000
        b = y[:, np.newaxis].copy()

        # this is what we wanna solve:
        #
        # min_w 1/2 ||y - X w||^2 + l1_reg ||w||_1 + 1/2 l2_reg ||w||^2
        #
        # w \in R^(feats x 1)
        f = lambda w : 0.5*(b-X.dot(w)).T.dot(b-X.dot(w)) \
                        + l1_reg*np.sum(np.abs(w)) + 0.5*l2_reg*w.T.dot(w)
        g = lambda u : 0.5*(b-X.dot(u)).T.dot(b-X.dot(u)) \
                        + 0.5*l2_reg*u.T.dot(u)

        tol = 1e-3
        w = start_pos[:, np.newaxis]
        tk = 1.
        beta = 0.5

        converged = False
        k = 0
        while not converged and max_iter > k:
            # optimization
            grad_w = X.T.dot(X.dot(w)) + l2_reg*w - X.T.dot(b)
            while True:
                z = prox_l1_operator(tk*l1_reg, w-tk*grad_w)
                if g(z) <= g(w) + grad_w.T.dot(z-w) + (1/(2*tk))*np.sum((z-w).T.dot(z-w)):
                    w = z
                    break
                tk *= beta   # reduce step length

            # convergence check
            obj = f(w)
            # check the dual gap first
            theta = (b - X.dot(w)) / l1_reg
            dual_obj = 0.5*b.T.dot(b) - 0.5*l1_reg*l1_reg*(theta-b/l1_reg).T.dot(theta-b/l1_reg) \
                       + 0.5*l2_reg*w.T.dot(w)
            # print k, ' - ', obj, dual_obj, np.abs(obj-dual_obj)
            if k > 0 and np.abs(obj-dual_obj) < tol:
                converged = True
            k += 1

        if debug:
            # baseline CD solver for comparison
            coefs, dual_gap, eps, n_iter_ = lm.cd_fast.enet_coordinate_descent(
                start_pos, l1_reg, l2_reg, np.asfortranarray(X), b.reshape((b.size)), max_iter, tol, np.random, 0, 0)
            print k, ' - ', np.abs(f(coefs[:, np.newaxis]) - f(w))
        print k

        return w.reshape((w.size)), 0, 0.


class AccelProximalGradientSolver(AbstractSolver):
    """ Proximal gradient solver for elastic net. """
    def __init__(self):
        AbstractSolver.__init__(self, 'AccelProxGradientSolver')

    def solve(self, start_pos, X, y, l1_reg, l2_reg, max_iter=20000, tol=1e-6, debug=False):
        # X \in R^(samples x feats)
        # y \in R^samples
        #max_iter = 20000
        b = y[:, np.newaxis]

        # print tol
        #tol = 1e-6

        # this is what we wanna solve:
        #
        # min_w 1/2 ||y - X w||^2 + l1_reg ||w||_1 + 1/2 l2_reg ||w||^2
        #
        # w \in R^(feats x 1)
        f = lambda w : 0.5*(b-X.dot(w)).T.dot(b-X.dot(w)) \
                        + l1_reg*np.sum(np.abs(w)) + 0.5*l2_reg*w.T.dot(w)
        g = lambda u : 0.5*(b-X.dot(u)).T.dot(b-X.dot(u)) \
                        + 0.5*l2_reg*u.T.dot(u)

        w = start_pos[:, np.newaxis]
        w_old = w
        tk = 1.
        beta = 0.5

        converged = False
        k = 0
        while not converged and max_iter > k:
            # optimization
            y = w + (k/(k+3))*(w - w_old)
            grad_y = X.T.dot(X.dot(y)) + l2_reg*y - X.T.dot(b)
            while True:
                z = prox_l1_operator(tk*l1_reg, w-tk*grad_y)
                if g(z) <= g(y) + grad_y.T.dot(z-y) + (1/(2*tk))*np.sum((z-y).T.dot(z-y)):
                    w_old = w
                    w = z
                    break
                tk *= beta   # reduce step length

            # convergence check
            obj = f(w)
            # check the dual gap first
            theta = (b - X.dot(w)) / l1_reg
            dual_obj = 0.5*b.T.dot(b) - 0.5*l1_reg*l1_reg*(theta-b/l1_reg).T.dot(theta-b/l1_reg) \
                       + 0.5*l2_reg*w.T.dot(w)
            # print k, ' - ', obj, dual_obj, np.abs(obj-dual_obj)
            if k > 0 and np.abs(obj-dual_obj) < tol:
                converged = True
            k += 1
        print k

        if debug:
            # baseline CD solver for comparison
            coefs, dual_gap, eps, n_iter_ = lm.cd_fast.enet_coordinate_descent(
                start_pos, l1_reg, l2_reg, np.asfortranarray(X), b.reshape((b.size)), max_iter, tol, np.random, 0, 0)
            print k, ' - ', np.abs(f(coefs[:, np.newaxis]) - f(w))
        return w.reshape((w.size)), 0, 0.


def prox_l1_operator(t, u):
    u_star1 = np.hstack([ u-t*np.ones((u.size, 1)), np.zeros((u.size, 1))])
    u_star2 = np.hstack([-u-t*np.ones((u.size, 1)), np.zeros((u.size, 1))])
    z = np.max(u_star1, axis=1) - np.max(u_star2, axis=1)
    return z[:, np.newaxis]