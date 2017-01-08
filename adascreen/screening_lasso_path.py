import numpy as np
import sklearn.linear_model as lm
import scipy.optimize as opt
import scipy.sparse as sparse
import sklearn as sk

from screening_elastic_net_path import ScreeningElasticNetPath

class ScreeningLassoPath(ScreeningElasticNetPath):
    """ Lasso solver with lambda path and screening. """

    def __init__(self, alg_screen, solver, path_lb=0.7, path_ub=1.0, path_steps=10, path_stepsize=0.9, path_scale='geometric'):
        ScreeningElasticNetPath.__init__(self, alg_screen, solver, path_lb, path_ub, path_steps, path_stepsize, path_scale)

    def fit(self, X, y, max_iter=20000, tol=1e-6, debug=True):
        return ScreeningElasticNetPath.fit(self, X, y, l2=0.0, tol=tol, debug=debug, max_iter=max_iter)
