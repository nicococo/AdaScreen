from numpy import zeros, array, eye
from numpy.random import multivariate_normal, uniform, random_integers

from sklearn.covariance import GraphLasso
import sklearn as skl
import matplotlib.pyplot as plt

from neighbor_select import NeighborSelect 
from solver import SklearnCDSolver, ActiveSetCDSolver, ProximalGradientSolver, AccelProximalGradientSolver
from screening_rules import *


if __name__ == '__main__':
    """ Sequential (standalone) version for testing graph-lasso
        implementation against scipy version. 
    """
    N = 200  # number of training examples
    DIMS = 5  # number of features
    COV = eye(DIMS)  # covariance matrix

    CORR_RANGE = [0.3, 0.6]  # min and max off-diagonal correlations
    CORR_FRAC = 0.4  # fraction of correlated features (0.0 <= CORR_FRAC < 1.0)

    # Well, this is not a very good way of generating a specific covariance matrix:
    # For certain parameters, it could result in negative semi-definite matrices.
    NUM_CORR = int(float(DIMS)*CORR_FRAC)
    for i in range(NUM_CORR):
        val = uniform(CORR_RANGE[0], CORR_RANGE[1])
        a = b = 0
        while a==b or COV[a,b]>0.0:
            (a,b) = random_integers(low=0, high=DIMS-1, size=2)
        COV[a,b] = val
        COV[b,a] = val

    # draw samples from a multivariate Gaussian
    X = multivariate_normal(mean=zeros(COV.shape[0]), cov=COV, size=N)
    X = skl.preprocessing.normalize(X, norm='l2').T

    # compute the empirical covariance matrix
    C_emp = X.dot(X.T)/float(N)
    print('Empirical Cov:')
    print C_emp

    # neighborhood selection
    nhs = NeighborSelect(EDPP(), AccelProximalGradientSolver(), path_lb=0.2, path_steps=5, path_scale='log')
    Cb = nhs.fit(np.ascontiguousarray(X))
    print Cb

    glasso = GraphLasso(alpha=0.005, tol=0.0001, max_iter=1000, verbose=False)
    glasso.fit(X.T)
    C = glasso.get_precision()
    print glasso.error_norm(COV)

    print('GraphLasso Cov:')
    print C

    # plot some example network
    plt.figure()
    plt.subplot(2, len(Cb), 1)
    plt.title('Cov')
    plt.pcolor(COV)
    plt.subplot(2, len(Cb), 2)
    plt.title('Emp. Cov')
    plt.pcolor(C_emp)
    plt.subplot(2, len(Cb), 3)
    plt.title('GraphLasso')
    plt.pcolor(C)

    for i in range(len(Cb)):
        plt.subplot(2, len(Cb), len(Cb)+1+i)
        foo = Cb[i].todense()
        foo[foo < 2.0] = 0.0
        plt.pcolor(np.array(foo))
    plt.show()

    print('Done.')

