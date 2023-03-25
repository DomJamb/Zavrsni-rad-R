import matplotlib.pyplot as plt
import numpy as np
import json

def show_loss(path, name, save=False, show=True):
    """
    Function for showcasing the loss over epochs
    Params:
        path: path to the needed statistics file
    """

    fig = plt.figure(figsize=(16, 10))

    with open(path, "r") as file:
        data = json.load(file)
    
    num_of_epochs = len(data)
    train_loss = list()
    test_loss = list()

    for key in data.keys():
        train_loss.append(data[key]["train_loss"])
        test_loss.append(data[key]["test_loss"])

    plt.plot(range(num_of_epochs), np.array(train_loss), "g-", label="Train loss")
    plt.plot(range(num_of_epochs), np.array(test_loss), "b-", label="Test loss")
    plt.xlabel("Epochs", labelpad=10, fontsize=12)
    plt.ylabel("Loss", labelpad=10, fontsize=12)
    plt.title("Train and test loss over the epochs")
    plt.legend(fontsize=12)

    if save:
        save_path = f"./stats/{name}.png"
        plt.savefig(save_path)

    if show:
        plt.show()

def show_accuracies(path, name, save=False, show=True):
    """
    Function for showcasing the train and test accuracy of a model over epochs
    Params:
        path: path to the needed statistics file
    """

    fig = plt.figure(figsize=(16,5))
    
    with open(path, "r") as file:
        data = json.load(file)
    
    num_of_epochs = len(data)
    train_acc = list()
    test_acc = list()

    for key in data.keys():
        train_acc.append(data[key]["train_accuracy"])
        test_acc.append(data[key]["test_accuracy"])

    plt.plot(range(num_of_epochs), np.array(train_acc), "g-", label="Train accuracy")
    plt.plot(range(num_of_epochs), np.array(test_acc), "b-", label="Test accuracy")

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Train and test accuracy over the epochs")
    plt.legend(fontsize=12)

    if save:
        save_path = f"./stats/{name}.png"
        plt.savefig(save_path)

    if show:
        plt.show()