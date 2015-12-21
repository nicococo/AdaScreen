import numpy as np

from screening_rules import AbstractScreeningRule

class Sasvi(AbstractScreeningRule):
    """ Screening by Sasvi rule. 
        Liu et al (2014) 
    """
    debug = False

    def __init__(self, tol=1e-9, debug=False):
        AbstractScreeningRule.__init__(self, 'Sasvi', tol=tol)
        self.debug = debug

    def screen(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        theta = (y - X[nz,:].T.dot(beta[nz])) / l0

        a = y/l0 - theta
        a_norm = np.linalg.norm(a) 
        a_norm2 = a_norm**2
        is_zero = ((a_norm/float(y.size)) < self.tol or self.isFirstIter(l0, lmax))

        b = a + (y/l - y/l0)
        b_norm = np.linalg.norm(b)

        Xtb = X.dot(b)
        XtO = X.dot(theta)

        if not is_zero:
            diff = (1.0/l - 1.0/l0) / 2.0
            Xta = X.dot(a)
        
            atb_normed = a.dot(b)/(a_norm*b_norm)
            yT = y - a/a_norm2 * a.dot(y)
            yT_norm = np.linalg.norm(yT)
            XT = X - Xta[:,np.newaxis].dot(a.reshape(1,a.size)/a_norm2) # feats x exms
            XT_norm = np.linalg.norm(XT, axis=1)
            XTtyT = XT.dot(yT)

            up =  XtO + diff*(XT_norm*yT_norm + XTtyT)  
            um = -XtO + diff*(XT_norm*yT_norm - XTtyT)  
            
            inds2 = np.where((Xta>+self.tol) & (atb_normed<=Xta/(normX*a_norm)))[0]
            inds3 = np.where((Xta<-self.tol) & (atb_normed<=-Xta/(normX*a_norm)))[0]

            um[inds2] = -XtO[inds2] + 0.5*(normX[inds2]*b_norm - Xtb[inds2])  
            up[inds3] =  XtO[inds3] + 0.5*(normX[inds3]*b_norm + Xtb[inds3])
        else:
            up =  XtO + 0.5*(normX*b_norm + Xtb)  
            um = -XtO + 0.5*(normX*b_norm - Xtb)  

        S = np.max((um, up), axis=0)
        inds = np.where(S >= 1.0 - self.tol)[0]

        # debug: compare with very slow but straightforward implementation
        if self.debug:
            (dbg_inds, dbg_interval) = self.screen_dbg(l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals)
            cinds = np.where(np.abs(inds-dbg_inds)>1e-6)[0]
            if cinds.size>0:
                print('Error: Implementations do not coincide.')
            else:
                print('You\'re fine..')

        return (inds, intervals)


    def screen_dbg(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        theta = (y - X[nz,:].T.dot(beta[nz])) / l0

        a = y/l0 - theta
        an = np.linalg.norm(a) 
        an2 = an*an
        is_zero = ((an/float(y.size)) < self.tol or self.isFirstIter(l0, lmax))

        b = a + (y/l - y/l0)
        bn = np.linalg.norm(b)

        diff = (1.0/l - 1.0/l0) / 2.0
        Xta = X.dot(a)

        if not is_zero:
            adotb = a.dot(b)/(an*bn)
            yT = y - a/an2 * a.dot(y)
            normyT = np.linalg.norm(yT)

        inds = []
        for j in range(X.shape[0]):
            xj = X[j,:]
            up = 10.0
            um = 10.0 
            # 1)
            if not is_zero and adotb>np.abs(Xta[j])/(normX[j]*an):
                xjT = xj - a/an2 * Xta[j]
                normxjT = np.linalg.norm(xjT)
                up =  theta.dot(xj) + diff*(normxjT*normyT + xjT.dot(yT))  
                um = -theta.dot(xj) + diff*(normxjT*normyT - xjT.dot(yT))  
            # 2)
            if not is_zero and Xta[j]>+self.tol and adotb<=Xta[j]/(normX[j]*an):
                xjT = xj - a/an2 * Xta[j]
                normxjT = np.linalg.norm(xjT)
                up =  theta.dot(xj) + diff*(normxjT*normyT + xjT.dot(yT))  
                um = -theta.dot(xj) + 0.5*(normX[j]*bn - xj.dot(b))  
            # 3)
            if not is_zero and Xta[j]<-self.tol and adotb<=-Xta[j]/(normX[j]*an):
                up = theta.dot(xj) + 0.5*(normX[j]*bn + xj.dot(b))  
                xjT = xj - a/an2 * Xta[j]
                normxjT = np.linalg.norm(xjT)
                um = -theta.dot(xj) + diff*(normxjT*normyT - xjT.dot(yT))  
            # 4)
            if is_zero:
                up =  theta.dot(xj) + 0.5*(normX[j]*bn + xj.dot(b))  
                um = -theta.dot(xj) + 0.5*(normX[j]*bn - xj.dot(b))  

            if um>=1.0-self.tol or up>=1.0-self.tol:
                inds.append(j)

        #print inds
        inds = np.array(inds).astype(int)
        return (inds, intervals)

    def get_sphere(self, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        E = (y - X[nz,:].T.dot(beta[nz])) / l0
        C = y/l

        o = 0.5*(E-C)+C
        rho = 0.5*np.linalg.norm(E-C)
        #rho = 0.5 * np.linalg.norm(X[nz,:].T.dot(beta[nz])/l0 + (y/l-y/l0) )
        #rho = np.sqrt(o.dot(o)-E.dot(C))
        return (o, rho)


    def get_local_halfspaces(self, o, l, l0, lmax, lmax_x, beta, X, y, normX, normy, nz, intervals):
        if not self.isFirstIter(l0, lmax):
            theta = (y - X[nz,:].T.dot(beta[nz])) / l0            
            ak = -(theta-y/l0)
            normak = np.linalg.norm(ak)
            bk = np.array([ak.dot(theta)])
            ak = ak.reshape(1, y.size)
        else:
            ak = np.array([])
            bk = np.array([])
            normak = np.array([])
        return (ak, bk, normak)
