import numpy as np
import matplotlib.pyplot as plt

num_layers = np.array([1, 2, 3, 4, 5, 6])
num_blocks = np.array([1, 2, 3])



"""
#validation accuracies
val_all_256 = np.array([86.25, 85.45, 87.48, 88.04, 93.01, 94.23])
val_all_256_err = np.array([1.54, 2.49, 2.03, 1.76, 3.88, 2.35])

val_flip_256 = np.array([86.67, 83.43, 92.81, 87.40, 93.61, 92.05])
val_flip_256_err = np.array([1.68, 2.23, 1.45, 1.08, 1.10, 5.76])

val_all_128 = np.array([86.92, 86.44, 87.08, 91.14, 93.29, 92.86])
val_all_128_err = np.array([4.14, 2.20, 2.32, 1.71, 2.09, 1.71])

val_flip_128 = np.array([89.93, 91.13, 92.80, 90.84, 95.41, 94.81])
val_flip_128_err = np.array([3.19, 3.97, 4.26, 3.07, 2.14, 1.42])

plt.title('Validation set accuracy of trained model vs number of convolutional layers')
plt.xlabel('Number of layers')
plt.ylabel('Percentage correct predictions')
plt.ylim(70, 100)
plt.errorbar(num_layers, val_all_256, yerr=val_all_256_err, marker='o', color='red')
plt.errorbar(num_layers, val_flip_256, yerr=val_flip_256_err, marker='s', color='blue')
plt.errorbar(num_layers, val_all_128, yerr=val_all_128_err,  marker='^', color='green')
plt.errorbar(num_layers, val_flip_128, yerr=val_flip_128_err,  marker='*', color='black')
#plt.legend(['all 256', 'flip 256', 'all_128', 'flip 128'], bbox_to_anchor=(0.3, -0.1))
plt.legend(['all 256', 'flip 256', 'all 128', 'flip 128'], loc='lower left')
plt.show()
plt.close()

#test accuracies
test_all_256 = np.array([90.74, 90.26, 78.70, 85.00, 82.12, 88.28])
test_all_256_err = np.array([0.81, 7.15, 0.45, 1.92, 0.29, 2.42])

test_flip_256 = np.array([91.09, 91.48, 85.65, 86.64, 90.89, 89.49])
test_flip_256_err = np.array([5.43, 4.21, 3.65, 5.36, 4.57, 5.01])

test_all_128 = np.array([87.46, 87.73, 85.11, 82.41, 78.20, 80.13])
test_all_128_err = np.array([3.42, 5.62, 7.35, 2.20, 1.68, 7.35])

test_flip_128 = np.array([90.68, 90.10, 90.66, 84.69, 88.74, 83.33])
test_flip_128_err = np.array([5.53, 3.69, 6.02, 1.80, 2.88, 2.62])

plt.title('Test set accuracy of trained model vs number of convolutional layers')
plt.xlabel('Number of layers')
plt.ylabel('Percentage correct predictions')
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