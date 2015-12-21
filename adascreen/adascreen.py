import numpy as np
import sklearn.linear_model as lm

from screening_rules import AbstractScreeningRule

    
class AdaScreen(AbstractScreeningRule):
    """ Adaptive Lasso Screening with halfspace constraints.  """

    sphere_rule = None # screening rule that produces sphere center (o) and radius (rho)
    local_hs_rules = None # list of local (=expensive) halfspace constraint returning screening rules
    global_hs_rules = None # list of global halfspace constraint returning screening rules

    A = None
    b = None
    normA = None

    debug = False

    def __init__(self, sphere_rule, tol=1e-9, debug=False):
        #AbstractScreeningRule.__init__(self, 'AdaScreen (o){0}'.format(sphere_rule.name), tol)
        AbstractScreeningRule.__init__(self, 'AdaScreen:(o){0}'.format(sphere_rule.name), tol)
        self.sphere_rule = sphere_rule
        self.local_hs_rules = []
        self.global_hs_rules = []
        self.debug = debug

    def add_local_hs_rule(self, rule):
        self.local_hs_rules.append(rule)
        self.name = '{0}+(/){1}'.format(self.name, rule.name)

    def add_global_hs_rule(self, rule):
        self.global_hs_rules.append(rule)
        self.name = '{0}+(/){1}'.format(self.name, rule.name)

    def init(self, lmax, lmax_x, X, y, normX, normy, path):
        print('AdaScreen initialize global halfspace constraints.')
        (self.A, self.b, self.normA) = self.get_global_halfspaces(lmax, lmax_x, X, y, normX, normy)

    def get_sphere(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        return self.sphere_rule.get_sphere(l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals)

    def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        (DIMS, EXMS) = X.shape
        (o, rho) = self.sphere_rule.get_sphere(l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals)

        # screening based on sphere constraint
        theta = (y - X[nz,:].T.dot(beta[nz])) / l0
        lhs = X.dot(o)
        rhs = 1.0 - normX*rho
        inds = np.where(np.abs(lhs) >= rhs-self.tol)[0]

        # if there are no constraints, then don't bother
        (A, b, normA) = self.get_local_halfspaces(o, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals)
        
        # (a) only local constraints or ..
        # (b) no constraints, no worries
        if b.size==0 & self.b.size==0:
            return (inds, intervals)
        # (c) only global constraints
        if b.size==0 & self.b.size>0:
            A = self.A
            b = self.b
            normA = self.normA
        else:
            # (d) mixed constraints
            if b.size>0 & self.b.size>0:
                A = np.concatenate((A, self.A))
                b = np.append(b, self.b)
                normA = np.append(normA, self.normA)

        # pre-calculations
        prod_jk = X[inds,:].dot(A.T)
        prod_ko = A.dot(o)

        # distance to origin for each hyperplane r \in K x 1
        r_k = (b - prod_ko) / normA # element-wise multiplication and division
        # change sign according to case
        r_inds = np.where(r_k >= 0.0)[0]
        r_mul = np.ones(b.size)
        r_mul[r_inds] = -1.0
        r_k = np.abs(r_k)

        #print 'Constraints x Datapoints {0}'.format(A.shape)
        cosines_alpha = prod_jk / (normX[inds].reshape(len(inds),1) * normA) # J x K
        sines_alpha = np.sqrt( np.maximum(1.0-cosines_alpha**2, 0.0) ) # J X K: the inner element-wise maximum(.) is due to numerics

        rhos_plus  = self.screen_inner(r_k, r_mul, rho,  cosines_alpha, sines_alpha)
        #rhos_plus  = self.screen_inner_dbg_icml(r_k, r_mul, rho,  cosines_alpha, sines_alpha, prod_ko, b)
        S_plus  =  lhs[inds] + normX[inds]*rhos_plus

        rhos_minus = self.screen_inner(r_k, r_mul, rho, -cosines_alpha, sines_alpha)
        #rhos_minus = self.screen_inner_dbg_icml(r_k, r_mul, rho, -cosines_alpha, sines_alpha, prod_ko, b)
        S_minus = -lhs[inds] + normX[inds]*rhos_minus

        S = np.max((S_plus, S_minus), axis=0)
        active = np.where(S >= 1.0 - self.tol)[0]

        #print inds.size-active.size
        if self.debug:
            #print 'AdaScreen DEBUG START'
            (prodjk_dbg, cos_dbg, sin_dbg) = self.cosines_dbg(X[inds,:], A, normX[inds], normA)
            (rows, cols) = np.where(np.abs(prodjk_dbg-prod_jk)>1e-6)
            if rows.size>0:
                print 'PROD_JK:'
                print (rows, cols)

            (rows, cols) = np.where(np.abs(cos_dbg-cosines_alpha)>1e-6)
            if rows.size>0:
                print 'COS_ALPHA:'
                print (rows, cols)

            (rows, cols) = np.where(np.abs(sin_dbg-sines_alpha)>1e-6)
            if rows.size>0:
                print 'SIN_ALPHA:'
                print (rows, cols)

                print normX
                print normy

            rhos_plus_dbg  = self.screen_inner_dbg(r_k, r_mul, rho,  cos_dbg, sin_dbg)
            rhos_minus_dbg = self.screen_inner_dbg(r_k, r_mul, rho, -cos_dbg, sin_dbg)

            #print 'AdaScreen DEBUG END'


        #raw_input("Press Enter to continue...")
        #rhos_min = np.min((rhos_plus, rhos_minus), axis=0)
        #active = np.where(np.abs(lhs[inds])>=1.0 - normX[inds]*rhos_min - self.tol)[0]
        return (inds[active], intervals)

    def cosines_dbg(self, X, A, normX, normA):
        prod_jk = np.zeros((X.shape[0], A.shape[0]))
        for j in range(X.shape[0]):
            for k in range(A.shape[0]):
                for n in range(A.shape[1]):
                    prod_jk[j,k] += X[j,n] * A[k,n]

        cos_alpha = np.zeros(prod_jk.shape)
        sin_alpha = np.zeros(prod_jk.shape)
        for j in range(prod_jk.shape[0]):
            for k in range(prod_jk.shape[1]):
                cos_alpha[j,k] = prod_jk[j,k] / (normX[j]*normA[k])
                sin_alpha[j,k] = np.sqrt( np.maximum(1.0 - cos_alpha[j,k]*cos_alpha[j,k], 0.))

        return (prod_jk, cos_alpha, sin_alpha)


    def screen_inner(self, r, r_mul, rho, cos_alpha, sin_alpha):
        rhos_prime = rho*np.ones(sin_alpha.shape) # J x K
        (rows, cols) = np.where(cos_alpha-r/rho>0.0)   

        if any(rho**2 - r[cols]**2)<0.0:
            print 'dsdfgdggds'
        values = np.maximum(rho**2 - (np.sqrt(rho**2 - r[cols]**2) * cos_alpha[rows, cols] + r_mul[cols]*r[cols]*sin_alpha[rows, cols])**2, 0.0)
        rhos_prime[rows, cols] = np.sqrt(values) 

        return np.min(rhos_prime, axis=1) # J x 1


    def screen_inner_dbg(self, r, r_mul, rho, cos_alpha, sin_alpha):
        (J, K) = sin_alpha.shape
        rhos_prime = rho*np.ones(sin_alpha.shape) # J x K
        for j in range(J):
            for k in range(K):
                if cos_alpha[j,k]>r[k]/rho:
                    #print (j,k)
                    #print rho**2 - (np.sqrt(rho**2 - r[k]**2) * cos_alpha[j,k] + r_mul[k]*r[k]*sin_alpha[j,k])**2
                    value = rho**2 - (np.sqrt(rho**2 - r[k]**2) * cos_alpha[j,k] + r_mul[k]*r[k]*sin_alpha[j,k])**2
                    if value<0.0:
                        print value
                        value = 0.0
                    rhos_prime[j,k] = np.sqrt(value)
        return np.min(rhos_prime, axis=1) # J x 1


    def screen_inner_dbg_icml(self, r, r_mul, rho, cos_alpha, sin_alpha, prod_ko, b):
        (J, K) = sin_alpha.shape

        #print prod_ko.shape
        #print b.shape
        #print b

        rhos_prime = rho*np.ones(sin_alpha.shape) # J x K
        for j in range(J):
            for k in range(K):
                                
                if (cos_alpha[j,k]>r[k]/rho and b[k]-prod_ko[k]>=0):
                    value = rho**2 - (np.sqrt(rho**2 - r[k]**2) * cos_alpha[j,k] + r_mul[k]*r[k]*sin_alpha[j,k])**2
                    if value<0.0:
                        print value
                        value = 0.0
                    rhos_prime[j,k] = np.sqrt(value)

                if (sin_alpha[j,k]<=r[k]/rho and b[k]-prod_ko[k]<0):
                    value = rho**2 - (np.sqrt(rho**2 - r[k]**2) * cos_alpha[j,k] + r_mul[k]*r[k]*sin_alpha[j,k])**2
                    if value<0.0:
                        print value
                        value = 0.0
                    rhos_prime[j,k] = np.sqrt(value)

                if (sin_alpha[j,k]>r[k]/rho and b[k]-prod_ko[k]<0):
                    value = np.sqrt(rho**2 - r[k]**2) * sin_alpha[j,k] - r[k]*cos_alpha[j,k]
                    rhos_prime[j,k] = value

        return np.min(rhos_prime, axis=1) # J x 1


    def get_local_halfspaces(self, o, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        # gather halfspace constraints
        A = None
        b = np.array([])
        normA = None
        doInit = True
        for rule in self.local_hs_rules:
            #print('Getting halfspace constraints of {0}..'.format(rule))
            (ak, bk, normak) = rule.get_local_halfspaces(o, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals)
            if doInit and ak.size>0:
                A = ak
                b = bk
                normA = normak
                doInit = False
            elif ak.size>0:
                A = np.concatenate((A, ak))
                b = np.append(b, bk)
                normA = np.append(normA, normak)
        # returns a_k, b_k and ||a_k||
        # A \in R^(K x N)
        # b \in R^K
        # normA \in R_+^K
        #print A.shape
        return (A, b, normA)

    def get_global_halfspaces(self, lmax, lmax_x, X, y, normX, normy):
        # gather halfspace constraints
        A = None
        b = np.array([])
        normA = None
        doInit = True
        for rule in self.global_hs_rules:
            #print('Getting halfspace constraints of {0}..'.format(rule))
            (ak, bk, normak) = rule.get_global_halfspaces(lmax, lmax_x, X, y, normX, normy)
            if doInit:
                A = ak
                b = bk
                normA = normak
                doInit = False
            else:
                A = np.concatenate((A, ak))
                b = np.append(b, bk)
                normA = np.append(normA, normak)
        # returns a_k, b_k and ||a_k||
        # A \in R^(K x N)
        # b \in R^K
        # normA \in R_+^K
        #print A.shape
        return (A, b, normA)
