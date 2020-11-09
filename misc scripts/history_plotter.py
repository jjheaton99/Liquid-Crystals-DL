import matplotlib.pyplot as plt

#takes training history and plots figure 
#displaying training and validation losses and accuracies
def plot_loss_acc_history(history):
    fig, axis = plt.subplots(2)
    fig.suptitle('Sequential model training losses and accuracies')
    
    axis[0].plot(history.history['loss'], label='loss')
    axis[0].plot(history.history['val_loss'], label='val_loss')
    axis[0].set_xlabel('Epoch')
    axis[0].set_ylabel('Loss')
    axis[0].legend(loc='upper right')
    
    axis[1].plot(history.history['accuracy'], label='accuracy')
    axis[1].plot(history.history['val_accuracy'], label='val_accuracy')
    axis[1].set_xlabel('Epoch')
    axis[1].set_ylabel('Accuracy')
    axis[1].legend(loc='lower right')

    plt.show()