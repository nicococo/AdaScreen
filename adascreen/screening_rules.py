import numpy as np


class AbstractScreeningRule(object):
    """ Base class for LASSO screening rules. """
    tol = 1e-9 # tolerance
    name = 'None' # name of associated lasso screening rule
    isComplete = True # does NOT screen out possibly non-zero coefficients

    def __init__(self, name, complete=True, tol=1e-9):
        self.tol = tol
        self.name = name
        self.isComplete = complete

    def isComplete(self):
        return self.isComplete

    def isFirstIter(self, l0, lmax):
        return abs(l0-lmax) < self.tol

    def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        # returns indices of non-screened features and (updated) intervals
        pass

    def get_local_halfspaces(self, o, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        raise NotImplementedError('Lasso Screening Rule {0} has no local halfspace constraints.'.format(self.name))

    def get_global_halfspaces(self, lmax, lmax_x, X, y, normX, normy):
        raise NotImplementedError('Lasso Screening Rule {0} has no global halfspace constraints.'.format(self.name))

    def init(self, lmax, lmax_x, X, y, normX, normy, path):
        # in case for cache etc..
        print('Screening ({0}): nothing to initialize.'.format(self.name))

    def release(self):
        # in case for cache etc..
        print('Screening ({0}): nothing to release.'.format(self.name))

    def __str__(self):
        return '{0}'.format(self.name)

    def __name__(self):
        return '{0}'.format(self.name)


class AbstractSphereScreeningRule(AbstractScreeningRule):

    def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        (o, r) = self.get_sphere(l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals)
        inds = np.where(np.abs(X.dot(o)) >= 1.0 - normX*r - self.tol)[0]
        return (inds, intervals)

    def get_sphere(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        raise NotImplementedError('Lasso Screening Rule {0} is not a sphere constraints.'.format(self.name))



class ScreenDummy(AbstractScreeningRule):
    """ No screening at all. """
    inds = None

    def __init__(self, tol=1e-9):
        AbstractScreeningRule.__init__(self, 'None', tol=tol)

    def init(self, lmax, lmax_x, X, y, normX, normy, path):
        self.inds = np.array(range(X.shape[0]))

    def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        return (self.inds, intervals)



class SAFE(AbstractSphereScreeningRule):
    """ El Ghaoui et al. (2010) """

    def __init__(self, tol=1e-9):
        AbstractScreeningRule.__init__(self, 'SAFE', tol=tol)

    #def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
    #    lhs = np.abs(X.dot(y))
    #    rhs = l - normX * normy * (lmax - l)/lmax         
    #    inds = np.where(lhs >= rhs - self.tol)[0]
    #    return (inds, intervals)

    def get_sphere(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        o = y/lmax
        rho = (1.0/l - 1.0/lmax)*normy
        return (o, rho)



class SeqSAFE(AbstractScreeningRule):
    """ El Ghaoui et al. (2010) """

    def __init__(self, tol=1e-9):
        AbstractScreeningRule.__init__(self, 'SeqSAFE', tol=tol)

    def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        theta0 = (y - X[nz,:].T.dot(beta[nz])) 
        a0 = theta0.T.dot(theta0)
        b0 = np.abs(y.T.dot(theta0))
        D = a0*np.max( (b0/a0 - l/l0), 0.0)**2.0 + y.T.dot(y) - (b0*b0)/a0
        rhs = np.abs(X.dot(y)) + np.sqrt(D)*normX
        inds = np.where(l < rhs - self.tol)[0]
        return (inds, intervals)


class DPP(AbstractSphereScreeningRule):
    """ Screening by Dual Polytope Projection """

    def __init__(self, tol=1e-9):
        AbstractScreeningRule.__init__(self, 'DPP', tol=tol)

    #def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
    #    # Original [d]ual [p]olytope [p]rojection
    #    # get the former dual solution
    #    theta = (y - X[nz,:].T.dot(beta[nz])) / l0
    #    # screen
    #    lhs = np.abs(X.dot(theta))
    #    mul = np.abs(1.0/l - 1.0/l0) * normy
    #    rhs = 1.0 - normX * mul
    #    inds = np.where(lhs >= rhs - self.tol)[0]
    #    return (inds, intervals)

    def get_sphere(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        # get the former dual solution
        o = (y - X[nz,:].T.dot(beta[nz])) / l0
        rho = np.abs(1.0/l - 1.0/l0) * normy
        return (o, rho)




class StrongRule(AbstractSphereScreeningRule):
    """ Tibshirani et al. (2012): Strong Rules for Discarding Predictors in Lasso-type Problems """

    def __init__(self, tol=1e-9):
        AbstractScreeningRule.__init__(self, 'Strong', complete=False, tol=tol)

    #def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
    #    theta = (y - X[nz,:].T.dot(beta[nz])) / l0
    #    lhs = np.abs(X.dot(theta))
    #    rhs = 2.0*l/l0 - 1.0
    #    inds = np.where(lhs >= rhs - self.tol)[0]
    #    return (inds, intervals)

    def get_sphere(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        theta = (y - X[nz,:].T.dot(beta[nz])) / l0
        rho = 2.0 * (1.0 - l/l0)
        #print l/l0
        return (theta, rho)


class DOME(AbstractScreeningRule):
    """ Screening by DOME rule. 
        Xiang and Ramadge (2012) 

        One-shot method
    """

    def __init__(self, tol=1e-9):
        AbstractScreeningRule.__init__(self, 'DOME', tol=tol)
        print('Warning! Dome-screening is *not* optimized.')

    def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        inds = []
        for i in range(X.shape[0]):
            t = X[i,:].dot(lmax_x)
            val = X[i,:].dot(y)

            r = np.sqrt(np.maximum(1.0/(lmax*lmax) - 1.0, 0.0)) * (lmax/l - 1.0)

            ql = self.__Ql(t, l, lmax, r)
            qu = self.__Qu(t, l, lmax, r)
            if ql>=val-self.tol or val>=qu-self.tol:
                inds.append(i)
        return (inds, intervals)

    def __Ql(self, t, l, lmax, r):
        if t<=lmax:
            if 1.0-t*t<0.0:
                #print 1.0-t*t
                return (lmax-l)*t - l
            return (lmax-l)*t - l + l*r*np.sqrt(1.0-t*t) 
        return -(l-1.0 + l/lmax)

    def __Qu(self, t, l, lmax, r):
        if t<-lmax:
            return (l-1.0 + l/lmax) 
        if 1.0-t*t<0.0:
            #print 1.0-t*t
            return (lmax-l)*t + l 
        return (lmax-l)*t + l - l*r*np.sqrt(1.0-t*t)


    def get_sphere(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        #q = y/l - (lmax/l - 1.0)*lmax_x
        #r = np.sqrt(1.0/(lmax*lmax) - 1.0) * (lmax/l - 1.0)
        q = y/lmax
        r = normy*(1.0/l - 1.0/lmax)
        # include normy in the sqrt? (like in the ST3 rule) ??
        return (q, r)

    def get_local_halfspaces(self, o, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        n0 = lmax_x
        norm_n0 = np.linalg.norm(n0)
        c0 = np.array([1.0])
        n0 = n0.reshape(1, len(y))
        return (n0, c0, norm_n0)

    def get_global_halfspaces(self, lmax, lmax_x, X, y, normX, normy):
        n0 = lmax_x
        norm_n0 = np.linalg.norm(n0)
        c0 = np.array([1.0])
        n0 = n0.reshape(1, len(y))
        return (n0, c0, norm_n0)



class ST3(DOME):
    """ Screening by ST3 rule. 
    """

    def __init__(self, tol=1e-9):
        AbstractScreeningRule.__init__(self, 'ST3', tol=tol)

    def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        raise NotImplementedError('ST3 standalone screening not implemented.')

    def get_sphere(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        theta_max = y/lmax
        o = theta_max - (lmax/l - 1.0)*lmax_x
        rho = np.sqrt((normy*normy)/(lmax*lmax) - 1.0) * (lmax/l - 1.0)
        return (o, rho)


class HSConstr(AbstractScreeningRule):
    max_constr = 10

    def __init__(self, max_constr=10, tol=1e-9):
        AbstractScreeningRule.__init__(self, 'DualLasso({0})'.format(max_constr), tol=tol)
        self.max_constr = max_constr

    def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        raise NotImplementedError('HSConstr only returns halfspace constraints.')

    def get_local_halfspaces(self, o, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        if not self.isFirstIter(l0, lmax):
            ak = np.concatenate((X[nz,:], -X[nz,:]))   
            bk = np.ones(2*nz.size)
            normak = np.linalg.norm(ak, axis=1)

            # find the 'max_constr' ak with the lowest value
            inds = np.argsort(normak[:len(nz)])
            inds = inds[:np.min([inds.size, self.max_constr])]
            inds = np.concatenate((inds, inds+len(nz)))

            ak = ak[inds,:]
            bk = bk[inds]
            normak = normak[inds] 
        else:
            ak = np.array([lmax_x, -lmax_x])   
            bk = np.ones(2)
            normak = np.linalg.norm(ak, axis=1)
        return (ak, bk, normak)



class IDPP(DPP):
    """ Interval Screening by Dual Polytope Projection """

    def __init__(self, tol=1e-9):
        AbstractScreeningRule.__init__(self, 'IDPP', tol)

    def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        # [i]nterval [d]ual [p]olytope [p]rojection
        # old dual solution
        theta = (y - X[nz,:].T.dot(beta[nz])) / l0

        # screen only active features
        minds = np.where(intervals >= l)[0]

        lhs = np.abs(X[minds,:].dot(theta))
        mul = self.normX[minds] * normy
        xminl = mul / (1.0 + mul / l0 - lhs - self.tol)

        # update intervals of active features 
        nintervals = np.array(intervals)
        nintervals[minds] = xminl
        
        # find violators within the active set
        inds = minds[np.where(xminl >= l)[0]]
        return (inds, nintervals)




class EDPP(AbstractSphereScreeningRule):
    """ Enhanced Screening by Dual Polytope Projection """

    def __init__(self, tol=1e-9):
        AbstractScreeningRule.__init__(self, 'EDPP', tol)

    #def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
    #    # [e]nhanced [d]ual [p]olytope [p]rojection
    #    # old dual solution
    #    theta = (y - X[nz,:].T.dot(beta[nz])) / l0
    #    v1 = y / l0 - theta
    #    if abs(lmax-l0) < self.tol:
    #        v1 = lmax_x * np.sign(lmax_x.T.dot(y))
    #    v2 = y/l - theta
    #    v2t = v2 - v1 * v1.T.dot(v2) / v1.T.dot(v1)
    #    lhs = np.abs(X.dot(theta + 0.5*v2t))
    #    rhs = 1.0 - 0.5*normX*np.linalg.norm(v2t)
    #    inds = np.where(lhs >= rhs - self.tol)[0]
    #    return (inds, intervals)
    
    def get_sphere(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        # get the former dual solution
        # old dual solution
        theta = (y - X[nz,:].T.dot(beta[nz])) / l0
        v1 = y / l0 - theta
        if abs(lmax-l0) < self.tol:
            v1 = lmax_x * np.sign(lmax_x.T.dot(y))
        v2 = y/l - theta
        v2t = v2 - v1 * v1.T.dot(v2) / v1.T.dot(v1)
        o = theta + 0.5*v2t
        rho = 0.5*np.linalg.norm(v2t)
        #print rho
        return (o, rho)



class IEDPP(EDPP):
    """ Interval Enhanced Screening by Dual Polytope Projection """

    def __init__(self, tol=1e-9):
        AbstractScreeningRule.__init__(self, 'IEDPP', tol)

    def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        # [e]nhanced [d]ual [p]olytope [p]rojection
        # old dual solution
        theta = (y - X[nz,:].T.dot(beta[nz])) / l0
        
        # screen active features
        minds = np.where(intervals >= l)[0]
        v1 = y/l0 - theta
        if abs(lmax-l0) < self.tol:
            v1 = lmax_x * np.sign(lmax_x.T.dot(y))

        def evaluate(t, idx, foo):
            v2 = y/t - theta
            v2t = v2 - v1 * v1.T.dot(v2) / v1.T.dot(v1)
            lhs = np.abs(X[idx,:].dot(theta + 0.5*v2t))
            rhs = 1.0 - 0.5*normsX[idx]*np.linalg.norm(v2t)
            return abs(rhs - lhs) # abs for better control of the error (instead of e.g. quadratic) 

        xminl = np.zeros(len(minds))
        for i in range(len(minds)):
            xminl[i] = opt.minimize_scalar(evaluate, args=(minds[i], 0 ), method='Brent', tol=self.tol/2.0, bracket=(l, l0)).x

        # update intervals of active features 
        nintervals = np.array(intervals)
        nintervals[minds] = xminl

        # find violators within the active set
        inds = minds[np.where(xminl >= par-self.tol)[0]]
        return (inds, intervals)

