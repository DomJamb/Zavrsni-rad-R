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

def compare_stats(name, model_name, save=False, show=True):
    """
    Function for comparing multiple accuracies and losses of a model
    Params:
        name: path
        model_name: desired output name
        save: option to save the image
        show: option to show the image
    """
    fig = plt.figure(figsize=(16, 10))

    path = f'./stats/{name}/stats.json'

    # Fetch data
    with open(path, "r") as file:
        data = json.load(file)

    num_of_epochs = len(data) - 1

    train_loss = list()
    train_acc = list()
    test_not_poisoned_acc = list()
    test_poisoned_acc = list()

    for key in data.keys():
        if key != "train_time":
            train_loss.append(data[key]["train_loss"])
            train_acc.append(data[key]["train_accuracy"])
            test_not_poisoned_acc.append(data[key]["test_accuracy"])
            test_poisoned_acc.append(data[key]["test_poisoned_accuracy"])

    # Plot the data
    plt.plot(range(1, num_of_epochs+1), np.array(train_loss), label="Train loss")
    plt.plot(range(1, num_of_epochs+1), np.array(train_acc), label="Train accuracy")
    plt.plot(range(1, num_of_epochs+1), np.array(test_not_poisoned_acc), label="Test accuracy, not poisoned data")
    plt.plot(range(1, num_of_epochs+1), np.array(test_poisoned_acc), label="Test accuracy, poisoned data")

    plt.xlabel("Epochs", labelpad=10, fontsize=12)
    plt.title(f"Stats comparison for model {model_name}", fontsize=20)
    plt.legend(fontsize=12)

    if save:
        save_path = f"./stats/{name}/stats_comparison.png"
        plt.savefig(save_path)

    if show:
        plt.show()

def compare_train_loss(names, save_name, save=False, show=True):
    """
    Function for showcasing the train loss over batches for multiple models
    Params:
        names: dict with paths and desired graph labels
        save_name: save path name
        save: option to save the image
        show: option to show the image
    """
    fig = plt.figure(figsize=(16, 10))

    for name in names.keys():
        # Get the loss from the specified input file
        path = f'./stats/{name}/train_loss.json'

        with open(path, "r") as file:
            data = json.load(file)

        # Plot the train loss over batches
        plt.plot(range(1, len(data)+1), np.array(data), label=f"{names[name]}")

    plt.xlabel("Batches", labelpad=10, fontsize=12)
    plt.ylabel("Loss", labelpad=10, fontsize=12)

    plt.title("Train loss over batches", fontsize=20)
    plt.legend(fontsize=12)

    if save:
        save_path = f"./stats/{save_name}.png"
        plt.savefig(save_path)

    if show:
        plt.show()

def compare_train_accs(names, save_name, save=False, show=True):
    """
    Function for showcasing the train accuracy over batches for multiple models
    Params:
        names: dict with paths and desired graph labels
        save_name: save path name
        save: option to save the image
        show: option to show the image
    """
    fig = plt.figure(figsize=(16, 10))

    for name in names.keys():
        # Get the accuracy from the specified input file
        path = f'./stats/{name}/train_accs.json'

        with open(path, "r") as file:
            data = json.load(file)

        # Plot the train accuracy over batches
        plt.plot(range(1, len(data)+1), np.array(data), label=f"{names[name]}")

    plt.xlabel("Batches", labelpad=10, fontsize=12)
    plt.ylabel("Accuracy", labelpad=10, fontsize=12)

    plt.title("Train accuracy over batches", fontsize=20)
    plt.legend(fontsize=12)

    if save:
        save_path = f"./stats/{save_name}.png"
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

def show_stats(name, save_name, save=False, show=True):
    """
    Function for showcasing the stats of various models
    Params:
        name: data path name
        save_name: save path name
        save: option to save the image
        show: option to show the image
    """
    path = f'./stats/{name}'

    names = list()
    test_accs = list()
    adv_accs = list()
    train_times = list()
    
    # Get the data from the specified input file
    with open(path, "r") as file:
        line = file.readline()
        while(line):
            if ("lr," in line):
                name = line.split(":")[0]
                names.append(name)
            elif ("Total test accuracy" in line):
                acc = line.split(": ")[1]
                test_accs.append(float(acc))
            elif ("20 steps" in line):
                acc = line.split(": ")[1]
                adv_accs.append(float(acc[0:len(acc)-2]))
            elif ("Total train time" in line):
                time = line.split("is ")[1]
                minutes = time.split(" minutes")[0]
                seconds = (time.split(", ")[1]).split(" seconds")[0]
                total_time = int(minutes) + int(seconds)/60
                train_times.append(total_time)
            line = file.readline()

    # Test accuracy plot
    fig1 = plt.figure(figsize=(16, 10))
 
    plt.bar(np.arange(len(test_accs)), test_accs, width=0.5, color="green")
    
    plt.xticks([r for r in range(len(test_accs))], names, rotation="vertical", fontsize=14)
    plt.ylabel('Test accuracy', fontsize=14, labelpad=20)
    plt.title("Test accuracy for various models", fontweight='bold', fontsize=20, pad=15)

    plt.subplots_adjust(bottom=0.25)

    if save:
        save_path = f"./stats/{save_name}_test_acc.png"
        plt.savefig(save_path)
    if show:
        plt.show()

    # Adversarial accuracy plot
    fig2 = plt.figure(figsize=(16, 10))
 
    plt.bar(np.arange(len(adv_accs)), adv_accs, width=0.5, color="orange")
    
    plt.xticks([r for r in range(len(adv_accs))], names, rotation="vertical", fontsize=14)
    plt.ylabel('Adversarial accuracy, PGD 20 steps', fontsize=14, labelpad=20)
    plt.title("Adversarial accuracy for various models", fontweight='bold', fontsize=20, pad=15)

    plt.subplots_adjust(bottom=0.25)

    if save:
        save_path = f"./stats/{save_name}_adv_acc.png"
        plt.savefig(save_path)
    if show:
        plt.show()

    # Train time plot
    fig3 = plt.figure(figsize=(16, 10))
 
    plt.bar(np.arange(len(train_times)), train_times, width=0.5, color="blue")
    
    plt.xticks([r for r in range(len(train_times))], names, rotation="vertical", fontsize=14)
    plt.ylabel('Train time', fontsize=14, labelpad=20)
    plt.title("Train time for various models", fontweight='bold', fontsize=20, pad=15)

    plt.subplots_adjust(bottom=0.25)

    if save:
        save_path = f"./stats/{save_name}_train_time.png"
        plt.savefig(save_path)
    if show:
        plt.show()

    # Test accuracy and adversarial accuracy comparison plot

    barWidth = 0.25
    fig4 = plt.subplots(figsize=(16, 10))
    
    br1 = np.arange(len(test_accs))
    br2 = [x + barWidth for x in br1]
    
    plt.bar(br1, test_accs, color ='green', width = barWidth,
            edgecolor ='grey', label ='Test accuracy')
    plt.bar(br2, adv_accs, color ='orange', width = barWidth,
            edgecolor ='grey', label ='Adversarial accuracy, PGD 20 steps')
    
    plt.xticks([r + barWidth for r in range(len(test_accs))], names, rotation="vertical", fontsize=14)
    plt.ylabel('Accuracy', fontsize=14, labelpad=20)
    plt.title("Test accuracy and adversarial accuracy comparison for various models", fontweight='bold', fontsize=20, pad=15)
    plt.legend()

    plt.subplots_adjust(bottom=0.25)
    
    if save:
        save_path = f"./stats/{save_name}.png"
        plt.savefig(save_path)

    if show:
        plt.show()