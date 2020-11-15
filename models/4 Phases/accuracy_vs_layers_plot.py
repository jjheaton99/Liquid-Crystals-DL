import numpy as np
import matplotlib.pyplot as plt

num_layers = np.array([1, 2, 3, 4, 5, 6])

#validation accuracies
val_acc_flip = np.array([86.40, 85.59, 94.33, 88.72, 92.65, 97.86])
val_acc_all = np.array([85.53, 82.46, 85.47, 89.75, 89.75, 93.86])

plt.title('Peak validation accuracy of trained model vs number of convolutional layers')
plt.xlabel('Number of layers')
plt.ylabel('Percentage correct predictions')
plt.plot(num_layers, val_acc_flip, marker='o', color='blue')
plt.plot(num_layers, val_acc_all, marker='x', color='red')
plt.legend(['flip augmentations only', 'all augmentations'], loc='lower right')
plt.show()
plt.close()

#test accuracies
test_acc_flip = np.array([86.71, 85.91, 87.86, 92.10, 90.15, 94.50])
test_acc_all = np.array([90.15, 95.55, 79.27, 87.06, 82.13, 86.37])

plt.title('Peak test accuracy of trained model vs number of convolutional layers')
plt.xlabel('Number of layers')
plt.ylabel('Percentage correct predictions')
plt.plot(num_layers, test_acc_flip, marker='o', color='blue')
plt.plot(num_layers, test_acc_all, marker='x', color='red')
plt.legend(['flip augmentations only', 'all augmentations'], loc='lower right')
plt.show()
plt.close()

#training times
training_time_flip = np.array([3040, 2180, 3040, 2289, 3108, 3990])
training_time_all = np.array([6210, 5520, 5750, 5474, 6486, 5290])

plt.title('Model training time vs number of convolutional layers')
plt.xlabel('Number of layers')
plt.ylabel('Training time (s)')
plt.plot(num_layers, training_time_all, marker='o', color='blue')
plt.plot(num_layers, training_time_flip, marker='x', color='red')
plt.legend(['flip augmentations only', 'all augmentations'], loc='center left')
plt.show()
plt.close()