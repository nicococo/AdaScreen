import numpy as np
import sklearn.linear_model as lm

from screening_rules import AbstractScreeningRule

    
class BagScreen(AbstractScreeningRule):
    """ Bagging of Multiple Lasso Screening Rules.  """

    rules = None # screening rules

    def __init__(self, rule, tol=1e-9, debug=False):
        AbstractScreeningRule.__init__(self, 'BagScreen:{0}'.format(rule.name), tol)
        self.rules = [rule]

    def add_rule(self, rule):
        self.rules.append(rule)
        self.name = '{0}+{1}'.format(self.name, rule.name)

    def init(self, lmax, lmax_x, X, y, normX, normy, path):
        print('BagScreen: nothing to initialize.')

    def get_sphere(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        raise NotImplementedError('Lasso Screening Rule {0} has no single sphere constraint.'.format(self.name))

    def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        (DIMS, EXMS) = X.shape
 
        inds = np.array(range(DIMS))
        for i in range(len(self.rules)):
            (linds, ints) = self.rules[i].screen(l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals)
            inds = np.intersect1d(inds,linds)

        return (inds, intervals)
