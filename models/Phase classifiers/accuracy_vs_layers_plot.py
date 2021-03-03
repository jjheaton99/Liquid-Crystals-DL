import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
import pandas as pd

plt.rcParams['axes.titley'] = 1.05
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

num_layers = np.array([1, 2, 3, 4, 5, 6])
num_blocks = np.array([0.85, 1, 2, 3, 3.15])

inc_val_accs = pd.read_csv('multi train results/smectic3/fl_inc_val_accs.csv').to_numpy()
inc_test_accs = pd.read_csv('multi train results/smectic3/fl_inc_test_accs.csv').to_numpy()

inc_val_mean = np.append(np.insert(inc_val_accs[12][1:], 0, 0), 0)
inc_val_err = np.append(np.insert(inc_val_accs[13][1:], 0, 0), 0)

inc_test_mean = np.append(np.insert(inc_test_accs[12][1:], 0, 0), 0)
inc_test_err = np.append(np.insert(inc_test_accs[13][1:], 0, 0), 0)

fig = plt.figure(figsize=(4.5, 4.5))
ax1 = fig.add_subplot()
ax1.set_title('Smectic 3 focal loss inception mean accuracies')
ax1.set_xlabel('Number of inception blocks')
ax1.set_ylabel('Mean accuracy in percent')
ax1.set_ylim(20, 100)
trans1 = ax1.transData + ScaledTranslation(-4/72, 0, fig.dpi_scale_trans)
trans2 = ax1.transData + ScaledTranslation(+4/72, 0, fig.dpi_scale_trans)
ax1.errorbar(num_blocks, inc_val_mean, yerr=inc_val_err, marker='o', linestyle='none', transform=trans1)
ax1.errorbar(num_blocks, inc_test_mean, yerr=inc_test_err, marker='s', linestyle='none', transform=trans2)
ax1.legend(['validation', 'test'], loc='lower right')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

#plt.tight_layout(w_pad=3.0, h_pad=2.0)
plt.show()
plt.close()

#Semester one plots
"""
inc_val_accs = pd.read_csv('multi train results/smecticAC/inc_val_accs.csv').to_numpy()
inc_test_accs = pd.read_csv('multi train results/smecticAC/inc_test_accs.csv').to_numpy()
seq_val_accs = pd.read_csv('multi train results/smecticAC/seq_val_accs.csv').to_numpy()
seq_test_accs = pd.read_csv('multi train results/smecticAC/seq_test_accs.csv').to_numpy()

inc_val_mean = np.append(np.insert(inc_val_accs[3][1:], 0, 0), 0)
inc_val_err = np.append(np.insert(inc_val_accs[4][1:], 0, 0), 0)

inc_test_mean = np.append(np.insert(inc_test_accs[3][1:], 0, 0), 0)
inc_test_err = np.append(np.insert(inc_test_accs[4][1:], 0, 0), 0)

seq_val_mean = seq_val_accs[3][1:]
seq_val_err = seq_val_accs[4][1:]

seq_test_mean = seq_test_accs[3][1:]
seq_test_err = seq_test_accs[4][1:]

fig = plt.figure(figsize=(8, 4.5))
#fig.suptitle('Mean accuracies for smectic A and C models', fontsize=19)
spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[2, 1])

ax1 = fig.add_subplot(spec[1])
#ax1.set_title('Inception models')
ax1.set_title('(b)')
ax1.set_xlabel('Number of inception blocks')
ax1.set_ylabel('Mean accuracy in percent')
ax1.set_ylim(85, 100)
trans1 = ax1.transData + ScaledTranslation(-4/72, 0, fig.dpi_scale_trans)
trans2 = ax1.transData + ScaledTranslation(+4/72, 0, fig.dpi_scale_trans)
ax1.errorbar(num_blocks, inc_val_mean, yerr=inc_val_err, marker='o', linestyle='none', transform=trans1)
ax1.errorbar(num_blocks, inc_test_mean, yerr=inc_test_err, marker='s', linestyle='none', transform=trans2)
ax1.legend(['validation', 'test'], loc='lower right')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

ax2 = fig.add_subplot(spec[0])
#ax2.set_title('Sequential models')
ax2.set_title('(a)')
ax2.set_xlabel('Number of convolutional layers')
ax2.set_ylabel('Mean accuracy in percent')
ax2.set_ylim(85, 100)
trans1 = ax2.transData + ScaledTranslation(-4/72, 0, fig.dpi_scale_trans)
trans2 = ax2.transData + ScaledTranslation(+4/72, 0, fig.dpi_scale_trans)
ax2.errorbar(num_layers, seq_val_mean, yerr=seq_val_err, marker='o', linestyle='none', transform=trans1)
ax2.errorbar(num_layers, seq_test_mean, yerr=seq_test_err, marker='s', linestyle='none', transform=trans2)
ax2.legend(['validation', 'test'], loc='lower right')

plt.tight_layout(w_pad=3.0, h_pad=2.0)
plt.show()
plt.close()

#validation accuracies
val_all_256 = np.array([86.25, 85.45, 87.48, 88.04, 93.01, 94.23])
val_all_256_err = np.array([1.54, 2.49, 2.03, 1.76, 3.88, 2.35])

val_flip_256 = np.array([86.67, 83.43, 92.81, 87.40, 93.61, 92.05])
val_flip_256_err = np.array([1.68, 2.23, 1.45, 1.08, 1.10, 5.76])

val_all_128 = np.array([86.92, 86.44, 87.08, 91.14, 93.29, 92.86])
val_all_128_err = np.array([4.14, 2.20, 2.32, 1.71, 2.09, 1.71])

val_flip_128 = np.array([89.93, 91.13, 92.80, 90.84, 95.41, 94.81])
val_flip_128_err = np.array([3.19, 3.97, 4.26, 3.07, 2.14, 1.42])

#test accuracies
test_all_256 = np.array([90.74, 90.26, 78.70, 85.00, 82.12, 88.28])
test_all_256_err = np.array([0.81, 7.15, 0.45, 1.92, 0.29, 2.42])

test_flip_256 = np.array([91.09, 91.48, 85.65, 86.64, 90.89, 89.49])
test_flip_256_err = np.array([5.43, 4.21, 3.65, 5.36, 4.57, 5.01])

test_all_128 = np.array([87.46, 87.73, 85.11, 82.41, 78.20, 80.13])
test_all_128_err = np.array([3.42, 5.62, 7.35, 2.20, 1.68, 7.35])

test_flip_128 = np.array([90.68, 90.10, 90.66, 84.69, 88.74, 83.33])
test_flip_128_err = np.array([5.53, 3.69, 6.02, 1.80, 2.88, 2.62])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(9.5, 8))
#fig.suptitle('Mean accuracies for 4-phase sequential models', fontsize=19)

#ax1.set_title('All augmentations, 256 x 256 input size')
ax1.set_title('(a)')
ax1.set_xlabel('Number of convolutional layers')
ax1.set_ylabel('Mean accuracy in percent')
ax1.set_ylim(70, 100)
trans1 = ax1.transData + ScaledTranslation(-4/72, 0, fig.dpi_scale_trans)
trans2 = ax1.transData + ScaledTranslation(+4/72, 0, fig.dpi_scale_trans)
ax1.errorbar(num_layers, val_all_256, yerr=val_all_256_err, marker='o', linestyle='none', transform=trans1)
ax1.errorbar(num_layers, test_all_256, yerr=test_all_256_err, marker='s', linestyle='none', transform=trans2)
ax1.legend(['validation', 'test'], loc='lower left')

#ax2.set_title('Flip augmentations, 256 x 256 input size')
ax2.set_title('(b)')
ax2.set_xlabel('Number of convolutional layers')
ax2.set_ylabel('Mean accuracy in percent')
ax2.set_ylim(70, 100)
trans1 = ax2.transData + ScaledTranslation(-4/72, 0, fig.dpi_scale_trans)
trans2 = ax2.transData + ScaledTranslation(+4/72, 0, fig.dpi_scale_trans)
ax2.errorbar(num_layers, val_flip_256, yerr=val_flip_256_err, marker='o', linestyle='none', transform=trans1)
ax2.errorbar(num_layers, test_flip_256, yerr=test_flip_256_err, marker='s', linestyle='none', transform=trans2)
ax2.legend(['validation', 'test'], loc='lower left')

#ax3.set_title('All augmentations, 128 x 128 input size')
ax3.set_title('(c)')
ax3.set_xlabel('Number of convolutional layers')
ax3.set_ylabel('Mean accuracy in percent')
ax3.set_ylim(70, 100)
trans1 = ax3.transData + ScaledTranslation(-4/72, 0, fig.dpi_scale_trans)
trans2 = ax3.transData + ScaledTranslation(+4/72, 0, fig.dpi_scale_trans)
ax3.errorbar(num_layers, val_all_128, yerr=val_all_128_err, marker='o', linestyle='none', transform=trans1)
ax3.errorbar(num_layers, test_all_128, yerr=test_all_128_err, marker='s', linestyle='none', transform=trans2)
ax3.legend(['validation', 'test'], loc='lower left')

#ax4.set_title('Flip augmentations, 128 x 128 input size')
ax4.set_title('(d)')
ax4.set_xlabel('Number of convolutional layers')
ax4.set_ylabel('Mean accuracy in percent')
ax4.set_ylim(70, 100)
trans1 = ax4.transData + ScaledTranslation(-4/72, 0, fig.dpi_scale_trans)
trans2 = ax4.transData + ScaledTranslation(+4/72, 0, fig.dpi_scale_trans)
ax4.errorbar(num_layers, val_flip_128, yerr=val_flip_128_err, marker='o', linestyle='none', transform=trans1)
ax4.errorbar(num_layers, test_flip_128, yerr=test_flip_128_err, marker='s', linestyle='none', transform=trans2)
ax4.legend(['validation', 'test'], loc='lower left')

plt.tight_layout(w_pad=3.0, h_pad=2.0)
plt.show()
plt.close()
"""
"""
plt.rcParams['axes.titley'] = 1.05
plt.title('Mean validation accuracies for 4-phase sequential models')
plt.xlabel('Number of convolutional layers')
plt.ylabel('Percentage accuracy')
plt.ylim(70, 100)
plt.errorbar(num_layers, val_all_256, yerr=val_all_256_err, marker='o', color='red')
plt.errorbar(num_layers, val_flip_256, yerr=val_flip_256_err, marker='s', color='blue')
plt.errorbar(num_layers, val_all_128, yerr=val_all_128_err,  marker='^', color='green')
plt.errorbar(num_layers, val_flip_128, yerr=val_flip_128_err,  marker='*', color='black')
#plt.legend(['all 256', 'flip 256', 'all_128', 'flip 128'], bbox_to_anchor=(0.3, -0.1))
plt.legend(['all 256', 'flip 256', 'all 128', 'flip 128'], loc='lower left')
plt.show()
plt.close()

plt.rcParams['axes.titley'] = 1.05
plt.title('Mean test accuracies for 4-phase sequential models')
plt.xlabel('Number of convolutional layers')
plt.ylabel('Percentage accuracy')
plt.ylim(70, 100)
plt.errorbar(num_layers, test_all_256, yerr=test_all_256_err, marker='o', color='red')
plt.errorbar(num_layers, test_flip_256, yerr=test_flip_256_err, marker='s', color='blue')
plt.errorbar(num_layers, test_all_128, yerr=test_all_128_err,  marker='^', color='green')
plt.errorbar(num_layers, test_flip_128, yerr=test_flip_128_err,  marker='*', color='black')
#plt.legend(['all 256', 'flip 256', 'all_128', 'flip 128'], bbox_to_anchor=(0.3, -0.1))
plt.legend(['all 256', 'flip 256', 'all 128', 'flip 128'], loc='lower left')
plt.show()
plt.close()
"""