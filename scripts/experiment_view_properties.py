import numpy as np



class ExperimentViewProperties(object):
    """ An instance of this class is returned for each experiment 
        and contains plotting information.  
    """
    title = 'experiment'
    info = ''
    x_axis_name = 'x-axis'
    y_axis_name = 'y-axis'
    xscale = 'linear'
    feats = 0
    exms = 0
    names = None
    legend_loc = 1
    marker = ['o','^','p','*','D','o','s','H','h','<','>']
    colors = ['b','g','r','c','m','y','k',np.random.rand(3),np.random.rand(3),np.random.rand(3)]

    def __init__(self, title, x, y, loc=1, xscale='linear'):
        self.x_axis_name = x
        self.y_axis_name = y
        self.title = title
        self.legend_loc = loc
        self.names = []
        self.xscale = xscale
        
    def setStats(self, X):
        (self.exms, self.feats) = X.shape

    def getMarker(self, i, line='-'):
        return '{0}{1}'.format(line, self.marker[i])

    def getColor(self, i, line='-'):
        if i<len(self.colors):
            return self.colors[i]
        return np.random.rand(3)

    def getFname(self):
		fname =  '{0}_{1}x{2}_{3}'.format(self.title.replace(' ' ,'_'),self.exms,self.feats,self.info)
		if len(self.names) == 1:
			fname = fname+str(self.names[0]).replace(' ' ,'_').replace('/',"-")
		return fname

    def getStats(self):
        return 'Features $ = {0}$\nExamples $ = {1}$'.format(self.feats, self.exms)

    def plot(self, x, means, stds, save_pdf=True, directory=''):
        self.show(x, means, stds, xscale='linear', nomarker=False, save_pdf=save_pdf, directory=directory)        
        self.show(x, means, stds, xscale='log', nomarker=False, save_pdf=save_pdf, directory=directory)       
        self.show(x, means, stds, xscale='linear', nomarker=True, save_pdf=save_pdf, directory=directory)        
        self.show(x, means, stds, xscale='log', nomarker=True, save_pdf=save_pdf, directory=directory)        
        self.show(x, means, stds, use_stds=False, xscale='linear', nomarker=True, save_pdf=save_pdf, directory=directory)
        self.show(x, means, stds, use_stds=False, xscale='log', nomarker=True, save_pdf=save_pdf, directory=directory)

    def show(self, x, means, stds, xscale='linear', use_stds=True, nomarker=False, save_pdf=True, directory=''):
        if save_pdf:
            import matplotlib as mpl
            mpl.use('Agg')
        import matplotlib.pyplot as plt

        if not use_stds:
            stds = np.zeros(means.shape)
        plt.figure()
        for i in range(means.shape[0]):
            if nomarker:
                plt.errorbar(x, means[i,:], yerr=stds[i,:], fmt=self.getMarker(i), color=self.getColor(i))
            else:
                plt.errorbar(x, means[i,:], yerr=stds[i,:], fmt='', color=self.getColor(i))

        plt.title('{0} {1}'.format(self.title,self.info), fontsize=22)
        plt.xlabel(self.x_axis_name, fontsize=18)
        plt.ylabel(self.y_axis_name, fontsize=18)
        if self.names is not None:
            if len(self.names) > 0:
                plt.legend(self.names, loc=self.legend_loc, fontsize=12)
        plt.xticks(x, fontsize=10)
        plt.xscale(xscale)
        #plt.text(0.12, 0.85, '{0}'.format(self.getStats()), fontsize = 12, color = 'k')
        if save_pdf:
            plt.savefig('{0}{1}_{2}_{3}_{4}.pdf'.format(directory, self.getFname(),
                                                        xscale, nomarker, use_stds), format='pdf')
        else:
            plt.show()

