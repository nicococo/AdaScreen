import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
import experiment_view_properties as p
import experiment_impl

foo = np.load('results_Pie/Pie_solver.npz')
means = foo['means']
stds = foo['stds']
x = foo['x']

print foo._files
print foo['results']

names = []
for i in range(len(experiment_impl.solver)):
    names.append(experiment_impl.solver[i])
names.append('Baseline')

v = p.ExperimentViewProperties('Title', '$\lambda/\lambda_{max}$', 'y-axis', loc=2, xscale='log')
v.names = names
v.show(x, means, stds, use_stds=False, nomarker=True, save_pdf=False, xscale='log')