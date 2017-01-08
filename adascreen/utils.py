import time
import numpy as np
import sklearn as skl
import csv
import scipy.sparse as sparse


def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        dtime = time.time() - startTime_for_tictoc
        print "Elapsed time is " + str(dtime) + " seconds."
        return dtime
    else:
        print "Toc: start time not set"
    return 0


def save_data(X, y, prefix='', delim=' '):
    # expect X \in M(EXMS x DIMS)
    np.savetxt('{0}_X.txt'.format(prefix), X, delimiter=delim)
    np.savetxt('{0}_y.txt'.format(prefix), y, delimiter=delim)


def load_svmlight_data(fname):
    (X, y) = skl.datasets.load_svmlight_file(fname)    
    print y.shape
    print X.shape
    print X
    return (X.todense(), y)


def load_simul_data(path):
    # load
    print('Load simul data.')
    X = np.loadtxt('{0}/simul_genome_data_X.txt'.format(path), delimiter=' ')
    y = np.loadtxt('{0}/simul_genome_data_Y.txt'.format(path), delimiter=' ')
    return (X, y)


def load_alzheimer_data(path):
    # load
    print('Load Alzheimer SNP data as 8 Bit integer.')
    X = np.loadtxt('{0}/genotype_X.txt'.format(path), delimiter=' ', dtype=np.uint8)    
    print X.shape
    print('Transform into float array.')
    X = X.astype(np.float_)
    print('Load Alzheimer phenotype data.')
    y = np.loadtxt('{0}/expression_Y.txt'.format(path), delimiter=' ')
    print y.shape
    num = np.random.randint(low=0, high=y.shape[1], size=1)
    print('Alzheimer dataset: choose random column from y = {0}.'.format(num))
    y = y[:, num]
    print('Size of data:')
    print X.nbytes
    print 'Done.'
    return (X, y[:,0])


def load_pems_data(path, num_feats=-1):
    # load
    print('Load PEMS data.')
    X = np.loadtxt('{0}/PEMS_X.txt'.format(path), delimiter=' ')
    y = np.loadtxt('{0}/PEMS_Y.txt'.format(path), delimiter=' ')
    print '-------'
    print y.shape
    print X.shape
    print '-------'

    if num_feats>0:
        print('PEMS data number of feature chosen by --toy_feats cmd line argument.')
        print('Select {0} feature of {1} by random.'.format(num_feats, X.shape[1]))
        inds = np.random.permutation(range(X.shape[1]))
        X = X[:,inds[:num_feats]]
        print X.shape
        print '-------'
   
    return (X, y)


def load_pie_data(path, transpose=False):
    # load
    print('Load PIE data.')
    X = np.loadtxt('{0}/PIE_data_X.txt'.format(path), delimiter=' ')
    y = np.loadtxt('{0}/PIE_data_y.txt'.format(path), delimiter=' ')
    print '-------'
    print y.shape
    print X.shape
    print '-------'

    if transpose:
        X = np.concatenate((X.T, y[:,np.newaxis].T))
        print X.shape
        num = np.random.randint(low=0, high=X.shape[1])
        print('PIE dataset: choosing idx={0} as response y.'.format(num))
        y = X[:,num]
        X = np.delete(X, num, 1)
        print '-------'
        print y.shape
        print X.shape
        print '-------'
    return (X, y)


def load_toy_data(exms=100, feats=10000, non_zeros=1000, sigma=0.1, corr=0.5):
    # Generate data similar as done in the Sasvi paper
    #X = np.random.randn(exms, feats)
    X = np.random.uniform(low=0.,high=+1., size=(exms, feats))
    for i in range(1,feats):
        X[:,i] = (1.0-corr)*X[:,i] + corr*X[:,i-1]
    # (exms x features)
    beta_star = np.random.uniform(low=-1., high=+1., size=feats)
    cut = feats-non_zeros
    inds = np.random.permutation(range(feats))
    beta_star[inds[:cut]] = 0.0
    y = X.dot(beta_star) + sigma*np.random.rand(exms)
    return (X, y)


def load_rand_data(exms=28, feats=10000):
    # see RAND-setting in Xiang et al. 'Screening Tests for Lasso Problems', 2014
    X = np.random.rand(exms, feats)
    y = np.random.rand(exms)
    return (X, y)


def load_sklearn_data(exms=100, feats=1000, n_informative=10):
    print dir(skl)
    (X, y, coef) = skl.datasets.make_regression(exms, feats, n_informative=n_informative, coef=True)
    print coef[np.abs(coef)>1e-12]
    return (X, y)


def load_GWAS_data(genotype_file, phenotype_file, covariates_file=None, exms=-1, permute=True):
    import pysnptools.pysnptools.util.util
    from pysnptools.pysnptools.snpreader.bed import Bed
    import fastlmm.util.standardizer as stdizer
    import fastlmm.pyplink.plink as plink

    snp_reader = Bed(genotype_file)
    phenotype = plink.loadOnePhen(phenotype_file)
    if covariates_file is not None:
        covariates = plink.loadPhen(covariates_file)
    else:
        covariates = None
    snp_reader, phenotype, covariates = pysnptools.pysnptools.util.util.intersect_apply([snp_reader, phenotype, covariates])
    
    if exms>0:#subset number individuals
        if permute:
            print snp_reader.iid_count
            perms = np.random.permutation(range(snp_reader.iid_count))
            np.savetxt('inds.txt', perms[0:exms])
            snp_reader = snp_reader[perms[0:exms],:]
        else:
            snp_reader=snp_reader[0:exms,:]
    #read the SNPs
    snp_data = snp_reader.read(order='C')

    X = snp_data.val
    if covariates is not None:
        X = np.hstack((covariates,X))
    stdizer.Unit().standardize(X)
    X.flags.writeable = False
    y = phenotype['vals'][:,0]

    print "done reading"
    return (X, y)


def normalize_data(X=None, y=None, mean_free=True):
    # expect X \in M(EXMS x DIMS)
    # (a) normalize y to have unit norm
    # (b) normalize X such that each feature has unit norm
    print('Normalizing data. X0={0}, X1={1}.'.format(X.shape[0], X.shape[1]))
    if X is not None:
        print('Calculate mean:')
        mX= np.mean(X, axis=0)
        print('Normalize using sklearn:')
        #Y = skl.preprocessing.normalize(X.T, norm='l2').T
        skl.preprocessing.normalize(X, norm='l2', axis=0, copy=False)
        #print np.diag(X.T.dot(X))
        #print np.diag(Y.T.dot(Y))
        if not mean_free:
            X += mX
    #X = skl.preprocessing.normalize(X, norm='l2')
    if y is not None:
        my = np.mean(y)
        y -= my
        #y /= np.sqrt(y.dot(y.T))
        y /= np.linalg.norm(y, ord=2)
        if not mean_free:
            y += my
    # return X \in M(EXMS x DIMS)
    print('Done.')
    return (X, y)
