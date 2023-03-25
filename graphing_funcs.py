import matplotlib.pyplot as plt
import numpy as np
import json

def show_loss(name, save=False, show=True):
    """
    Function for showcasing the loss over epochs
    Params:
        name: name of the model
        save: option to save the image
        show: option to show the image
    """

    fig = plt.figure(figsize=(16, 10))

    # Get the loss from the specified input file
    path = f'./stats/{name}/stats.json'

    with open(path, "r") as file:
        data = json.load(file)
    
    num_of_epochs = len(data)-1
    train_loss = list()
    test_loss = list()

    for key in data.keys():
        if key != "train_time":
            train_loss.append(data[key]["train_loss"])
            test_loss.append(data[key]["test_loss"])

    # Plot the train and test loss over epochs
    plt.plot(range(num_of_epochs), np.array(train_loss), "g-", label="Train loss")
    plt.plot(range(num_of_epochs), np.array(test_loss), "b-", label="Test loss")

    plt.xlabel("Epochs", labelpad=10, fontsize=12)
    plt.ylabel("Loss", labelpad=10, fontsize=12)

    plt.title("Train and test loss over the epochs")
    plt.legend(fontsize=12)

    if save:
        save_path = f"./stats/{name}/loss.png"
        plt.savefig(save_path)

    if show:
        plt.show()

def show_accuracies(name, save=False, show=True):
    """
    Function for showcasing the train and test accuracy of a model over epochs
    Params:
        name: name of the model
        save: option to save the image
        show: option to show the image
    """

    fig = plt.figure(figsize=(16, 10))

    path = f'./stats/{name}/stats.json'
    
    # Get the accuracies from the specified input file
    with open(path, "r") as file:
        data = json.load(file)
    
    num_of_epochs = len(data)-1
    train_acc = list()
    test_acc = list()

    for key in data.keys():
        if key != "train_time":
            train_acc.append(data[key]["train_accuracy"])
            test_acc.append(data[key]["test_accuracy"])

    # Plot the train and test accuracies over epochs
    plt.plot(range(num_of_epochs), np.array(train_acc), "g-", label="Train accuracy")
    plt.plot(range(num_of_epochs), np.array(test_acc), "b-", label="Test accuracy")

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)

    plt.title("Train and test accuracy over the epochs")
    plt.legend(fontsize=12)

    if save:
        save_path = f"./stats/{name}/accuracies.png"
        plt.savefig(save_path)

    if show:
        plt.show()