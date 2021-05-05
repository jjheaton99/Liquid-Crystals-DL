import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

def plot_best_models(val_accs, val_errs, test_accs, test_errs, 
                     lower_bound=50, upper_bound=100, linestyle='none'):
    plt.rcParams['axes.titley'] = 1.05
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12
    
    labels = ['Seq', 'Inc', 'RN50']
    
    fig = plt.figure(figsize=(3, 5))
    ax1 = fig.add_subplot()
    ax1.set_ylabel('Mean accuracy in percent')
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(lower_bound, upper_bound)
    trans1 = ax1.transData + ScaledTranslation(-4/72, 0, fig.dpi_scale_trans)
    trans2 = ax1.transData + ScaledTranslation(+4/72, 0, fig.dpi_scale_trans)
    ax1.errorbar(labels, 
                 val_accs, 
                 yerr=val_errs, 
                 marker='o', 
                 linestyle=linestyle, 
                 transform=trans1)
    ax1.errorbar(labels, 
                 test_accs, 
                 yerr=test_errs, 
                 marker='s', 
                 linestyle=linestyle, 
                 transform=trans2)
    ax1.legend(['validation', 'test'], loc='lower left')

    plt.show()