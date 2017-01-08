import numpy as np
import resource
from sklearn.metrics import mean_squared_error

from adascreen.solver import *
from adascreen.screening_rules import ScreenDummy
from adascreen.screening_lasso_path import ScreeningLassoPath
from experiment_view_properties import ExperimentViewProperties


# X-axis: lambda / lambda_max, Y-axis: rejection rate (one shot)
def screening_performance_one_shot(X, y, steps=65, screening_rules=None, solver_ind=0, geomul=0.9):
    return _screening_rejection_rate(X, y, True, steps=steps, screening_rules=screening_rules, solver_ind=solver_ind, geomul=geomul)


# X-axis: lambda / lambda_max, Y-axis: rejection rate (sequential screening)
def screening_performance_sequential(X, y, steps=65, screening_rules=None, solver_ind=0, geomul=0.9):
    return _screening_rejection_rate(X, y, False, steps=steps, screening_rules=screening_rules, solver_ind=solver_ind, geomul=geomul)


# X-axis: lambda / lambda_max, Y-axis: time in seconds
def path_times(X, y, steps=65, screening_rules=None, solver_ind=0, geomul=0.9):
    return _screening_times(X, y, steps=steps, solver_ind=solver_ind, speed_up=False,
                            screening_rules=screening_rules, geomul=geomul)


# X-axis: lambda / lambda_max, Y-axis: Speed-up
def path_speed_up(X, y, steps=65, screening_rules=None, solver_ind=0, geomul=0.9):
    return _screening_times(X, y, steps=steps, solver_ind=solver_ind, speed_up=True,
                            screening_rules=screening_rules, geomul=geomul)


# X-axis: lambda / lambda_max, Y-axis: time in seconds
def path_solver_acceleration(X, y, steps=65, screening_rules=None, solver_ind=-1, geomul=0.9):
    return _screening_solver_acceleration(X, y, steps=steps, screening_rules=screening_rules, geomul=geomul)


# X-axis: lambda / lambda_max, Y-axis: time in seconds
def path_solver_acceleration_nopath(X, y, steps=65, screening_rules=None, solver_ind=-1, geomul=0.9):
    return _screening_solver_acceleration(X, y, steps=steps, use_path_solver=False, screening_rules=screening_rules, geomul=geomul)


# X-axis: lambda / lambda_max, Y-axis: MSE
def path_accuracy(X, y, steps=65, screening_rules=None, solver_ind=-1, geomul=0.9):
    return _path_accuracy(X, y, steps=steps, train_test_ratio=0.8, geomul=geomul)


#  --------------------------------------------------------------------------------
#  EXPERIMENTS IMPLEMENTATION

# solver = [SklearnCDSolver(), SklearnLarsSolver(), ActiveSetCDSolver(0),
#           ActiveSetCDSolver(1), ProximalGradientSolver(), AccelProximalGradientSolver()]
solver = [SklearnCDSolver(), SklearnLarsSolver(),
          ProximalGradientSolver(), AccelProximalGradientSolver()]


def _screening_rejection_rate(X, y, one_shot, steps, screening_rules=None,
                              save_intermediate_results=True, solver_ind=1, geomul=0.9):
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
    myPath /= myPath[0]
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return myPath, res[:,:len(scr_inds)], props


def _screening_times(X, y, steps, solver_ind=None, speed_up=False, lower_bound=0.001, screening_rules=None, geomul=0.9):
    # X \in M(EXMS x DIMS)
    input = np.zeros(steps)
    props = ExperimentViewProperties('Runtime Comparison', '$\lambda / \lambda_{max}$', 'Time in [sec]', loc=1, xscale='log')
    res = np.zeros((len(screening_rules)+1, steps))  # all screening rules + solver w/o screening
    if speed_up:
        props = ExperimentViewProperties('Speed-up Comparison', '$\lambda / \lambda_{max}$', 'Speed-up', loc=1, xscale='log')
        res = np.ones((len(screening_rules)+1, steps))  # all screening rules
    props.setStats(X)

    for s in range(len(screening_rules)):
        print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        curr_solver_ind = solver_ind
        props.names.append(screening_rules[s])
        myLasso = ScreeningLassoPath(screening_rules[s], solver[curr_solver_ind], path_lb=lower_bound, path_steps=steps, path_stepsize=geomul, path_scale='geometric')
        beta, nz_inds, scr_inds, path, t1, t2 = myLasso.fit(X.T, y, tol=1e-4, debug=False)
        times = (np.array(t1) + np.array(t2)).tolist()
        if speed_up:  # do not consider screening code time
            times = (np.array(t1)).tolist()
        for i in range(1, steps):
            res[s, i] = float(np.sum(times[:i]))

    myLasso = ScreeningLassoPath(ScreenDummy(), solver[solver_ind], path_lb=lower_bound, path_steps=steps, path_stepsize=geomul, path_scale='geometric')
    beta, nz_inds, scr_inds, path, times1, _ = myLasso.fit(X.T, y, max_iter=1000, tol=1e-4, debug=False)
    props.names.append('Path solver w/o screening')

    input[0] = 1.0
    for i in range(1, len(path)):
        res_time = float(np.sum(times1[:i]))  # solver time w/o screening
        res[-1, i] = res_time
        if speed_up:
            res[:, i] = res_time / res[:, i]
        input[i] = path[i]/path[0]  # lambda / lambda_max
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return input, res, props


def _screening_solver_acceleration(X, y, steps, lower_bound=0.001, use_path_solver=True, screening_rules=None, geomul=0.9):
    if use_path_solver:
        props = ExperimentViewProperties('Path-solver Comparison', '$\lambda / \lambda_{max}$', 'Speed-up', loc=1, xscale='log')
    else:
        props = ExperimentViewProperties('Solver Comparison', '$\lambda / \lambda_{max}$', 'Speed-up', loc=1, xscale='log')
    props.setStats(X)

    input = np.zeros(steps)
    res = np.ones((len(solver)+1, steps))

    for s in range(len(solver)):
        print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        curr_solver_ind = s
        props.names.append(solver[s])

        myLasso = ScreeningLassoPath(screening_rules[0], solver[curr_solver_ind], path_lb=lower_bound, path_steps=steps, path_stepsize=geomul, path_scale='geometric')
        (beta, nz_inds, scr_inds, path, t1, t2) = myLasso.fit(X.T, y, tol=1e-2, max_iter=2000, debug=False)
        # times = (np.array(t1) + np.array(t2)).tolist()
        times = (np.array(t1)).tolist()  # do not consider screening algorithm time here
        for i in range(1, steps):
            res[s,i] = float(np.sum(times[:i]))

        myLasso = ScreeningLassoPath(ScreenDummy(), solver[curr_solver_ind], path_lb=lower_bound, path_steps=steps, path_stepsize=geomul, path_scale='geometric')
        if use_path_solver:
            beta, nz_inds, scr_inds, path, t1, _ = myLasso.fit(X.T, y, max_iter=2000, tol=1e-2, debug=False)
            times = (np.array(t1)).tolist()
            for i in range(1, steps):
                res[s, i] = float(np.sum(times[:i])) / res[s, i]
        else:
            myLasso.use_warm_start = False
            beta, nz_inds, scr_inds, path, t1, _ = myLasso.fit(X.T, y, max_iter=2000, tol=1e-2, debug=False)
            times = (np.array(t1)).tolist()
            for i in range(1, steps):
                res[s, i] = float(times[i-1]) / res[s, i]

    props.names.append('Solver w/o screening')

    input[0] = 1.0
    for i in range(1, len(path)):
        input[i] = path[i]/path[0]  # lambda / lambda_max

    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return input, res, props


def _path_accuracy(X, y, steps, train_test_ratio, geomul, lower_bound=0.001):
    props = ExperimentViewProperties('Accuracy', '$\lambda / \lambda_{max}$', 'MSE', loc=1, xscale='log')
    props.setStats(X)

    inds = np.random.permutation(X.shape[0])
    end = np.int(train_test_ratio*inds.size)
    X_train = X[inds[:end], :]
    X_test = X[inds[end:], :]
    y_train = y[inds[:end]]
    y_test = y[inds[end:]]

    myLasso = ScreeningLassoPath(ScreenDummy(), solver[0], path_lb=lower_bound, path_steps=steps, path_stepsize=geomul, path_scale='geometric')
    beta, nz_inds, scr_inds, path, _, _ = myLasso.fit(X_train.T, y_train, max_iter=1000, tol=1e-4, debug=False)
    y_preds = myLasso.predict_path(beta, X_test.T)
    print y_preds.shape

    res = np.zeros((1, steps))
    for i in range(steps):
        res[0, i] = mean_squared_error(y_test, y_preds[:, i])

    input = np.zeros(steps)
    input[0] = 1.0
    for i in range(1, len(path)):
        input[i] = path[i]/path[0]  # lambda / lambda_max

    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return input, res, props
