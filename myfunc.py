import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Plot one image with it's label
def plot_image(image, label, pred = None):
    plt.imshow(image)
    # Show true and predicted classes.
    if pred is None:
        xlabel = "True: {0}".format(label)
    else:
        xlabel = "True: {0}, Pred: {1}".format(label, pred)
    print(xlabel)
    
# Plot 9 images
def plot_9_images(images, labels, preds=None):
    assert len(images) == len(preds) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i])

        # Show true and predicted classes.
        if preds is None:
            xlabel = "True: {0}".format(labels[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(labels[i], preds[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
# Show 9 examples of incorrect predictions
def plot_example_errors(images, labels, preds):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (preds != labels)
    incorrect = [i[0] for i in incorrect]
    # Get the images from the test-set that have been
    # incorrectly classified.
    wrong_images = images[incorrect]
    
    # Get the predicted classes for those images.
    wrong_preds = preds[incorrect]

    # Get the true classes for those images.
    wrong_labels = labels[incorrect]
    
    # Plot the first 9 images.
    plot_9_images(wrong_images[0:9], wrong_labels[0:9], wrong_preds[0:9])
    
# Return one batch out of the dataset
def fetch_batch(images, labels, batch_size):
    num_images = len(images)

    # Create a random index.
    idx = np.random.choice(num_images, size=batch_size, replace=False)

    # Use the random index to select random x and y-values.
    # We use the transfer-values instead of images as x-values.
    x_batch = images[idx]
    y_batch = labels[idx]

    return x_batch, y_batch


#Confusion matrix   
def plot_confusion_matrix(cls_pred, cls_true):
    # This is called from print_test_accuracy() below.
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, range(2))
    plt.yticks(tick_marks, range(2))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()