import numpy as np
import matplotlib.pyplot as plt

num_layers = np.array([1, 2, 3, 4, 5, 6])
accuracy = np.array([86.40, 85.59, 94.33, 84.14, 92.65, 97.86])

plt.title('Peak validation accuracy of trained model vs number of convolutional layers')
plt.xlabel('Number of layers')
plt.ylabel('Percentage correct predictions')
plt.plot(num_layers, accuracy, marker='o')
plt.show()