import sys
import os
import argparse

from gridmap import Job, process_jobs


""" --------------------------------------------------------------------------------
    EXPERIMENTS
"""


def screening_performance_one_shot(X, y, steps=65, screening_rules=None, solver_ind=1, geomul=0.9):
    return screening_rejection_rate(X, y, True, steps=steps, screening_rules=screening_rules, solver_ind=solver_ind, geomul=geomul)


def screening_performance_sequential(X, y, steps=65, screening_rules=None, solver_ind=1, geomul=0.9):
    return screening_rejection_rate(X, y, False, steps=steps, screening_rules=screening_rules, solver_ind=solver_ind, geomul=geomul)


def path_times(X, y, steps=65, screening_rules=None, solver_ind=1, geomul=0.9):
    return path_over_time(X, y, steps=steps, solver_ind=solver_ind, screening_rules=screening_rules, geomul=geomul)


""" -------------------------------------------------------------------------------- 
    EXPERIMENTS IMPL
"""


def screening_rejection_rate(X, y, one_shot, steps, screening_rules=None, save_intermediate_results=True, solver_ind=1, geomul=0.9):
    import numpy as np
    import resource
    from adascreen.solver import SklearnCDSolver, SklearnLarsSolver, ActiveSetCDSolver
    from adascreen.screening_lasso_path import ScreeningLassoPath
    from experiment_view_properties import ExperimentViewProperties
    solver = [SklearnCDSolver(), SklearnLarsSolver(), ActiveSetCDSolver(0), ActiveSetCDSolver(1)]
    
    # X \in M(EXMS x DIMS)
    (EXMS, DIMS) = X.shape
    # plotting options
    setting = '(sequential)'
    if one_shot:
        setting = '(one shot)'
    props = ExperimentViewProperties('Screening Performance {0}'.format(setting), '$\lambda / \lambda_{max}$', 'Rejection Rate', 4, xscale='log')
    props.setStats(X)
    res = np.zeros((len(screening_rules), steps))
    for s in range(len(screening_rules)):
        print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss    

        props.names.append(screening_rules[s])
        myLasso = ScreeningLassoPath(screening_rules[s], solver[solver_ind], path_lb=0.001, path_stepsize=geomul, path_steps=steps, path_scale='geometric')
        #myLasso = ScreeningLassoPath(screening_rules[s], None, path_lb=0.05, path_steps=STEPS, path_scale='linear')
        myLasso.one_shot_screening = one_shot
        (beta, nz_inds, scr_inds, path, _, _) = myLasso.fit(X.T, y, tol=1e-4, debug=False)
        myPath = path
        for i in range(len(scr_inds)):
            res[s,i] = float(DIMS-len(scr_inds[i]))/float(DIMS)
        # if save_intermediate_results:
        #     props_intermediate = ExperimentViewProperties('Intermediate Screening Performance {0}'.format(setting), '$\lambda / \lambda_{max}$', 'Rejection Rate', 4, xscale='log')
        #     props_intermediate.setStats(X)
        #     props_intermediate.names.append(screening_rules[s])
        #     np.savez('{0}{1}'.format(directory,props_intermediate.getFname()), reps=arguments.reps, dataset=arguments.dataset, \
        #         nexms=EXMS, x=myPath/myPath[0], results=res[s:s+1,:])
    myPath /= myPath[0]
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss    

    return myPath, res[:,:len(scr_inds)], props


def path_over_time(X, y, steps, solver_ind=None, lower_bound=0.001, add_screening_time=True, screening_rules=None, geomul=0.9):
    import numpy as np
    import resource
    from adascreen.screening_rules import ScreenDummy
    from adascreen.solver import SklearnCDSolver, SklearnLarsSolver, ActiveSetCDSolver
    from adascreen.screening_lasso_path import ScreeningLassoPath
    from experiment_view_properties import ExperimentViewProperties
    solver = [SklearnCDSolver(), SklearnLarsSolver(), ActiveSetCDSolver(0), ActiveSetCDSolver(1)]

    # X \in M(EXMS x DIMS)
    (EXMS, DIMS) = X.shape
    props = ExperimentViewProperties('Runtime Comparison', '$\lambda / \lambda_{max}$', 'Time in [sec]', loc=1, xscale='log')
    props.setStats(X)

    input = np.zeros(steps)
    res = np.zeros((len(screening_rules)+1, steps))

    for s in range(len(screening_rules)):
        print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss    
        props.names.append(screening_rules[s])
        myLasso = ScreeningLassoPath(screening_rules[s], solver[solver_ind], path_lb=lower_bound, path_steps=steps, path_stepsize=geomul, path_scale='geometric')
        (beta, nz_inds, scr_inds, path, t1, t2) = myLasso.fit(X.T, y, tol=1e-4, debug=False)            
        times = (np.array(t1) + np.array(t2)).tolist()
        for i in range(1,steps):
            res[s,i] = float(np.sum(times[:i]))

    props.names.append('Path solver w/o screening')
    myLasso = ScreeningLassoPath(ScreenDummy(), solver[solver_ind], path_lb=lower_bound, path_steps=steps, path_stepsize=geomul, path_scale='geometric')
    (beta, nz_inds, scr_inds, path, times1, _) = myLasso.fit(X.T, y, max_iter=1000, tol=1e-4, debug=False)            

    input[0] = 1.0
    for i in range(1,len(path)):
        res[-1,i] = float(np.sum(times1[:i]))
        input[i] = path[i]/path[0]

    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss    
    return input, res, props


""" --------------------------------------------------------------------------------
"""


def remote_iteration(r, arguments, exms_to_load, directory):
    import numpy as np
    import resource
    from screening_rules import EDPP, DOME, DPP, SAFE, StrongRule, HSConstr
    from adascreen.bagscreen import BagScreen
    from adascreen.adascreen import AdaScreen
    from adascreen.sasvi import Sasvi
    import utils
    import uuid
    
    startMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss    
    
    # set a random seed
    np.random.seed(int(uuid.uuid1(r).int % 0xFFFFFFFF))

    ada_hs = AdaScreen(EDPP())
    ada_hs.add_local_hs_rule(HSConstr(max_constr=100))

    ada_sasvi = AdaScreen(EDPP())
    ada_sasvi.add_local_hs_rule(Sasvi())

    ada_sasvi1 = AdaScreen(Sasvi())
    ada_sasvi1.add_local_hs_rule(Sasvi())

    ada_dome = AdaScreen(EDPP())
    ada_dome.add_local_hs_rule(DOME())

    ada_full = AdaScreen(EDPP())
    ada_full.add_local_hs_rule(Sasvi())
    ada_full.add_local_hs_rule(HSConstr(max_constr=100))

    ada_strong = AdaScreen(StrongRule())
    ada_strong.add_local_hs_rule(Sasvi())
    ada_strong.add_local_hs_rule(HSConstr(max_constr=10))

    bag = BagScreen(EDPP())
    bag.add_rule(DOME())
    #bag.add_rule(Sasvi())

    ada_bag = AdaScreen(EDPP())
    ada_bag.add_global_hs_rule(DOME())
    #ada_bag.add_local_hs_rule(Sasvi())

    ada_dome1 = AdaScreen(DOME())
    ada_dome1.add_global_hs_rule(DOME())

    ada1 = AdaScreen(EDPP())
    ada1.add_local_hs_rule(HSConstr(max_constr=1))
    ada2 = AdaScreen(EDPP())
    ada2.add_local_hs_rule(HSConstr(max_constr=5))
    ada3 = AdaScreen(EDPP())
    ada3.add_local_hs_rule(HSConstr(max_constr=10))
    ada4 = AdaScreen(EDPP())
    ada4.add_local_hs_rule(HSConstr(max_constr=100))
    ada5 = AdaScreen(EDPP())
    ada5.add_local_hs_rule(HSConstr(max_constr=1000))

    ruleset_dome = [ada_dome1, DOME()]
    ruleset_sasvi = [ada_sasvi, Sasvi()]
    ruleset_short = [ada_full, Sasvi()]

    ###########################
    # Toy experiment settings #
    ###########################
    ruleset_all = [ada_full, ada_hs, ada_sasvi, Sasvi(), EDPP(), DPP(), DOME(), SAFE()]
    ruleset_bag = [ada_bag, bag, EDPP(), DOME(), Sasvi()]
    ruleset_strong = [ada_strong, StrongRule(), Sasvi(), EDPP()]
    ruleset_hsconstr = [ada1, ada2, ada3, ada4, ada5, EDPP()]

    EXPERIMENT_LIST = [ \
        screening_performance_one_shot,\
        screening_performance_sequential,\
        path_times, \
        ]

    print('\n*******************************************')
    print('Starting iteration {0}...'.format(r))
    print('*******************************************\n')
    # data path
    dpath = arguments.path

    # select set of screening rules
    screening_rules = eval('ruleset_{0}'.format(arguments.screening_rule_set))
    print('Chosen screening rule set:')
    for si in screening_rules:
        print('  {0}'.format(si))
    # optionally overwrite the global screening_rules by a single rule
    if arguments.screening_rule >= 0:
        screening_rules = [screening_rules[arguments.screening_rule]]

    # select solver
    use_solver = arguments.use_solver_ind
    print('Use the following solver: {0}'.format(use_solver))

    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss    

    # reload data every iteration (=either generate new data or choose another output y)
    if arguments.dataset=='Toy':
        (X, y) = utils.load_toy_data(exms=exms_to_load, feats=arguments.toy_feats, non_zeros=100, sigma=0.1, corr=arguments.corr)
    elif arguments.dataset=='Rand':
        (X, y) = utils.load_rand_data(exms=exms_to_load, feats=arguments.toy_feats)
    elif arguments.dataset=='Regression':
        (X, y) = utils.load_sklearn_data(exms=exms_to_load, feats=1000)
    elif arguments.dataset=='Pie':
        (X, y) = utils.load_pie_data('{0}Pie'.format(dpath), transpose=True)
        # 1024(feats) x 11553(exms)
    elif arguments.dataset=='Pems':
        (X, y) = utils.load_pems_data('{0}pems'.format(dpath), num_feats=arguments.toy_feats)
    elif arguments.dataset=='Alzheimer':
        (X, y) = utils.load_alzheimer_data('{0}alzheimer'.format(dpath))
    elif arguments.dataset=='Simul':
        (X, y) = utils.load_simul_data('{0}Simul'.format(dpath))
    elif arguments.dataset=='10K':
        (X, y) = utils.load_svmlight_data('{0}E2006.train'.format(dpath))
    elif arguments.dataset=='Year':
        (X, y) = utils.load_svmlight_data('{0}YearPredictionMSD'.format(dpath))
    elif arguments.dataset=='Leu':
        (X, y) = utils.load_svmlight_data('{0}leu'.format(dpath))
    elif arguments.dataset=='ARIC':
        (X, y) = utils.load_GWAS_data(genotype_file = r"\\erg00\Genetics\dbgap\ARIC\all", phenotype_file = r"\\erg00\Genetics\dbgap\ARIC\all-bmi01.phe",exms=exms_to_load)
    else:
        raise NotImplementedError("unkownd dataset");
    print('Data loaded. Exms={0} Feats={1}'.format(X.shape[0], X.shape[1]))
    print X.shape
    print y.shape

    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss    

    # if hold out set is enabled:
    if arguments.hold_out>0.01:
        in_exms = np.floor((1.0-arguments.hold_out)*y.size)
        print('Hold out set enabled. Using {0} of {1} examples.'.format(in_exms,y.size))
        perms = np.random.permutation(range(y.size))
        X = X[perms[:in_exms],:]
        y = y[perms[:in_exms]]
        print X.shape
        print y.shape

    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss    
    # normalize data
    (X, y) = utils.normalize_data(X, y, mean_free=True)
    # do the experiment

    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss    
    (x, res, props) = EXPERIMENT_LIST[arguments.experiment](X, y, solver_ind=use_solver, screening_rules=screening_rules, steps=arguments.steps, geomul=arguments.geometric_mul)
    # save the result

    print x 
    print res
    print res.shape
    output = 'results_{0}/intermediate/{1}_'.format(arguments.dataset, arguments.screening_rule_set)
    if arguments.dataset=='Toy':
        props.info = '(corr={0})'.format(arguments.corr)

    np.savez('{0}_run{1}_{2}'.format(output, r, props.getFname()), reps=arguments.reps, dataset=arguments.dataset, \
        nexms=X.shape[0], x=x, results=[res], means=res, stds=np.zeros(res.shape), arguments=arguments)
    props.plot(x, res, np.zeros(res.shape), save_pdf=True, directory='{0}_run{1}_'.format(output,r))    
    print('Experiment arguments:')
    print arguments
    print('Iteration {0} done.\n\n'.format(r))
    stopMem = startMem - resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Memory usage = {0} KB'.format(stopMem))
    return (res, props, x)


def remote_save_result(results, props, x, arguments, exms_to_load, directory):
    import numpy as np
    from experiment_view_properties import ExperimentViewProperties
    # combine all results
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0, ddof=0)
    print x 
    print results
    print means
    print stds
    if arguments.dataset=='Toy':
        props.info = '(corr={0})'.format(arguments.corr)
    np.savez('{0}{1}_{2}'.format(directory, \
        arguments.screening_rule_set, props.getFname()), reps=arguments.reps, \
        dataset=arguments.dataset, nexms=exms_to_load, x=x, results=results, \
        means=means, stds=stds, arguments=arguments)
    props.plot(x, means, stds, save_pdf=True, directory='{0}{1}_'.format(directory, \
        arguments.screening_rule_set))    



if __name__ == '__main__':
    import logging
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'), level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset", help="Dataset to run (default Toy)", default="Toy", type=str)
    parser.add_argument("-o","--hold_out", help="Fraction of hold-out examples for reps (default 0.0)", default=0.0, type =float)
    parser.add_argument("-r","--reps", help="number repetitions (default 1)", default=1, type =int)
    parser.add_argument("-e","--experiment", help="active experiment [0-2] (default 1)", default=1, type =int)
    parser.add_argument("-s","--screening_rule", help="active screening rule [0-3] (default -1=all)", default=-1, type =int)
    parser.add_argument("-l","--screening_rule_set", help="Select a screening rule set by name (default all)", default='all', type =str)
    parser.add_argument("-i","--steps", help="number of steps (default 65)", default=65, type =int)
    parser.add_argument("-g","--geometric_mul", help="Multiplier for geometric path (default 0.9)", default=0.9, type =float)
    parser.add_argument("-u","--use_solver_ind", help="Select the index of the solver to use (default 1 = sklearn LARS)", default=0, type =int)
    parser.add_argument("-p","--path", help="dataset path (default '/home/nicococo/Data/')", default='/home/nicococo/Data/', type =str)
    parser.add_argument("-c","--corr", help="Correlation coefficient for Toy dataset (default 0.6)", default=0.9, type =float)
    parser.add_argument("-t","--toy_exms", help="Number of toy examples (default 100)", default=25, type =int)
    parser.add_argument("-f","--toy_feats", help="Number of toy features (default 10000)", default=1000, type =int)
    parser.add_argument("-z","--mem_max", help="Ensures that processes do not need more than this amount of memory(default 16G)", default='16G', type =str)
    parser.add_argument("-m","--max_processes", help="Maximum number of processes (-1 = cluster) (default 1)", default=1, type =int)
    arguments = parser.parse_args(sys.argv[1:])

    print('Parameters:')
    print arguments
    print

    # this is for simulated data only
    exms_to_load = arguments.toy_exms

    # output directory
    directory = 'results_{0}'.format(arguments.dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = 'results_{0}/intermediate'.format(arguments.dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = 'results_{0}/'.format(arguments.dataset, arguments.screening_rule_set)

    # create empty job vector
    jobs = []
    for r in range(arguments.reps):
        job = Job(remote_iteration, [r, arguments, exms_to_load, directory], \
            mem_max=arguments.mem_max, mem_free='16G', name='{0}({1})'.format(arguments.dataset, arguments.screening_rule_set)) 
        jobs.append(job)
        #(res, props, x) = remote_iteration(r, arguments, exms_to_load, screening_rules, solver, directory)

    processedJobs = process_jobs(jobs, local=arguments.max_processes>=1, max_processes=arguments.max_processes)
    results = []
    print "ret fields AFTER execution on local machine"
    for (i, result) in enumerate(processedJobs):
        print "Job #", i
        (res, props, x) = result
        results.append(res)

    job = Job(remote_save_result, [results, props, x, arguments, exms_to_load, directory]) 
    processedJobs = process_jobs([job], local=arguments.max_processes>=1, max_processes=1)
    print "Saved results."

    print('Experiment arguments:')
    print arguments
    print
    print('Done.')
