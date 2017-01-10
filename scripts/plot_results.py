import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
import experiment_view_properties as p
import experiment_impl

from adascreen.screening_rules import EDPP, DOME, DPP, SAFE, StrongRule, HSConstr
from adascreen.bagscreen import BagScreen
from adascreen.adascreen import AdaScreen
from adascreen.sasvi import Sasvi

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
ruleset_base = [EDPP(), DPP(), DOME()]

foo = np.load('/Users/nicococo/Documents/AdaScreen/2/Alzheimer_Solver.npz')
# foo = np.load('/Users/nicococo/Documents/AdaScreen/2/Alzheimer_Speedup.npz')
# foo = np.load('/Users/nicococo/Documents/AdaScreen/2/Pems_Solver.npz')

# foo = np.load('/Users/nicococo/Documents/AdaScreen/Pems_all_Spe_440x138672_.npz')
# foo = np.load('/Users/nicococo/Documents/AdaScreen/Alzheimer_all_Spe_540x511997_.npz')
# foo = np.load('/Users/nicococo/Documents/AdaScreen/Pie_all_Spe_11554x1023_.npz')
means = foo['means']
stds = foo['stds']
# names = foo['props']
# print names.item().names
# names = names.item().names
x = foo['x']

IS_SPEEDUP = False
IS_SOLVER = True
X_OFFSET = 0

print foo._files
print foo['results']

if IS_SPEEDUP:
    names = []
    for i in range(len(ruleset_all)):
        names.append(ruleset_all[i])
    names.append('Solver w/o screening')
    means[-1, :] = 1.0

if IS_SOLVER:
    names = []
    for i in range(len(experiment_impl.solver)):
        names.append(experiment_impl.solver[i])
    names.append('Baseline')

print x
print means[0, :]
v = p.ExperimentViewProperties('Title', '$\lambda/\lambda_{max}$', 'y-axis', loc=2, xscale='log')
v.names = names
v.show(x[X_OFFSET:], means[:,X_OFFSET:], stds[:,X_OFFSET:], use_stds=False, nomarker=True, save_pdf=False, xscale='log')