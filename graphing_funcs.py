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
    plt.plot(range(1, num_of_epochs+1), np.array(train_loss), "g-", label="Train loss")
    plt.plot(range(1, num_of_epochs+1), np.array(test_loss), "b-", label="Test loss")

    plt.xlabel("Epochs", labelpad=10, fontsize=12)
    plt.ylabel("Loss", labelpad=10, fontsize=12)

    plt.title("Train and test loss over the epochs")
    plt.legend(fontsize=12)

    if save:
        save_path = f"./stats/{name}/loss.png"
        plt.savefig(save_path)

    if show:
        plt.show()

def show_train_loss(name, save=False, show=True):
    """
    Function for showcasing the train loss over batches
    Params:
        name: name of the model
        save: option to save the image
        show: option to show the image
    """
    fig = plt.figure(figsize=(16, 10))

    # Get the loss from the specified input file
    path = f'./stats/{name}/train_loss.json'

    with open(path, "r") as file:
        data = json.load(file)

    # Plot the train loss over batches
    plt.plot(range(1, len(data)+1), np.array(data), "g-", label="Train loss")

    plt.xlabel("Batches", labelpad=10, fontsize=12)
    plt.ylabel("Loss", labelpad=10, fontsize=12)

    plt.title("Train loss over batches")
    plt.legend(fontsize=12)

    if save:
        save_path = f"./stats/{name}/train_loss_batches.png"
        plt.savefig(save_path)

    if show:
        plt.show()

def show_train_accs(name, save=False, show=True):
    """
    Function for showcasing the train accuracies over batches
    Params:
        name: name of the model
        save: option to save the image
        show: option to show the image
    """
    fig = plt.figure(figsize=(16, 10))

    # Get the accuracies from the specified input file
    path = f'./stats/{name}/train_accs.json'

    with open(path, "r") as file:
        data = json.load(file)

    # Plot the train accuracy over batches
    plt.plot(range(1, len(data)+1), np.array(data), "g-", label="Train accuracy")

    plt.xlabel("Batches", labelpad=10, fontsize=12)
    plt.ylabel("Accuracy", labelpad=10, fontsize=12)

    plt.title("Train accuracy over batches")
    plt.legend(fontsize=12)

    if save:
        save_path = f"./stats/{name}/train_acc_batches.png"
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
    plt.plot(range(1, num_of_epochs+1), np.array(train_acc), "g-", label="Train accuracy")
    plt.plot(range(1, num_of_epochs+1), np.array(test_acc), "b-", label="Test accuracy")

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)

    plt.title("Train and test accuracy over the epochs")
    plt.legend(fontsize=12)

    if save:
        save_path = f"./stats/{name}/accuracies.png"
        plt.savefig(save_path)

    if show:
        plt.show()

def show_adversarial_accuracies_varying_steps(accs, name, save=False, show=True):
    """
    Function for showcasing the adversarial accuracy of a model over varying steps
    Params:
        accs: dict of accuracies over steps
        name: name of the model
        save: option to save the image
        show: option to show the image
    """
    fig = plt.figure(figsize=(16, 10))

    steps = list(accs.keys())
    vals = list(accs.values())

    # Plot the adversarial accuracies over varying number of PGD steps
    plt.plot(np.array(steps), np.array(vals), "g-", label="Adversarial accuracy")

    plt.xlabel("Number of PGD steps", fontsize=12)
    plt.ylabel("Accuracy on adversarial examples", fontsize=12)

    plt.title("Accuracy on adversarial examples over varying number of PGD steps")
    plt.legend(fontsize=12)

    if save:
        save_path = f"./stats/{name}/adv_accuracies_varying_steps.png"
        plt.savefig(save_path)

    if show:
        plt.show()

def show_adversarial_accuracies(name, save=False, show=True):
    """
    Function for showcasing the adversarial accuracy of a model over epochs
    Params:
        name: name of the model
        save: option to save the image
        show: option to show the image
    """
    fig = plt.figure(figsize=(16, 10))

    path = f'./stats/{name}/stats.json'
    
    # Get the adversarial accuracies from the specified input file
    with open(path, "r") as file:
        data = json.load(file)
    
    num_of_epochs = len(data)-1
    adv_acc = list()

    for key in data.keys():
        if key != "train_time":
            adv_acc.append(data[key]["adv_accuracy"])

    # Plot the train and test accuracies over epochs
    plt.plot(range(1, num_of_epochs+1), np.array(adv_acc), "g-", label="Adversarial accuracy")

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy on adversarial examples", fontsize=12)

    plt.title("Accuracy on adversarial examples over the epochs")
    plt.legend(fontsize=12)

    if save:
        save_path = f"./stats/{name}/adv_accuracies.png"
        plt.savefig(save_path)

    if show:
        plt.show()

def graph_adv_examples(adv_dict, name, save=False, show=True):
    """
    Function for showcasing the adversarial examples of a given model
    Params:
        adv_dict: dict[epoch: list<advExample>]
        name: name of the model
        save: option to save the image
        show: option to show the image
    """
    fig = plt.figure(figsize=(20,10))
    length = len(adv_dict.keys())
    keys = list(adv_dict.keys())

    subfigs = fig.subfigures(nrows=length, ncols=1)
    if not isinstance(subfigs, np.ndarray):
        subfigs = [subfigs]

    # Show adversarial examples for each correct label
    for row, subfig in enumerate(subfigs):
        key = keys[row]
        adv_list = adv_dict[key]
        adv_cnt = len(adv_list)
        subfig.suptitle(f'Epoch: {key}', fontweight='bold')

        axs = subfig.subplots(nrows=1, ncols=adv_cnt)
        i = 0

        for adv in adv_list:
            ax = axs[i]
            ax.plot()
            display_img = adv.attacked_img.transpose((1,2,0))
            ax.imshow(display_img)
            ax.set_title(f"Image class: {adv.img_class}")
            ax.axis('off')
            i += 1

    plt.subplots_adjust(top=0.75)

    if save:
        save_path = f"./stats/{name}/adv_examples.png"
        plt.savefig(save_path)

    if show:
        plt.show()