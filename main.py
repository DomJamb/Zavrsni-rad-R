import os
import json
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from attack_funcs import attack_pgd
from ResidualNetwork18 import ResidualNetwork18
from graphing_funcs import show_loss, show_accuracies
from util import get_train_time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(num_of_epochs, name):
    """
    Train function for the model initialized in the main function
    Params:
        name: desired name for the model (used for saving the model parameters in a JSON file)
    """
    start_time = time.time()
    train_stats = dict()
    for epoch in range(num_of_epochs):
        print(f"Starting epoch: {epoch + 1}")

        model.train()

        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            y_ = model(x)
            loss = loss_calc(y_, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, y_ = y_.max(1)
            train_total += y.size(0)
            train_correct += y_.eq(y).sum().item()

        total_train_acc = 100 * train_correct/train_total

        print(f"Total train loss for epoch {epoch+1}: {total_train_loss}")
        print(f"Total train accuracy for epoch {epoch+1}: {total_train_acc}")

        test_loss, test_acc = test(epoch)

        curr_epoch = f"epoch{epoch+1}"
        curr_dict = dict()
        curr_dict.update({"train_loss": total_train_loss, 
                          "train_accuracy": total_train_acc,
                          "test_loss": test_loss,
                          "test_accuracy": test_acc})
        
        train_stats.update({curr_epoch: curr_dict})

    total_time = time.time() - start_time
    train_stats.update({"train_time": total_time})

    file_path = f"./stats/{name}/stats.json"

    if (not os.path.exists(f"./stats/{name}")):
        os.mkdir(f"./stats/{name}")

    with open(file_path, "w") as file:
        json.dump(train_stats, file)

def train_free(num_of_epochs, name, replay=4):
    """
    Train function for the model initialized in the main function (implements Free Adversarial Training)
    Params:
        name: desired name for the model (used for saving the model parameters in a JSON file)
        replay: number of replays for each batch during 1 epoch
    """
    start_time = time.time()
    train_stats = dict()

    tensor_size = [train_loader.batch_size, 3, 32, 32]
    perturbation = torch.zeros(*tensor_size).to(device)

    num_of_epochs = math.ceil(num_of_epochs/replay)
    koef_it = 0.05
    eps = 0.3

    for epoch in range(num_of_epochs):
        print(f"Starting epoch: {epoch + 1}")

        model.train()

        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            temp_loss = 0
            temp_correct = 0

            for j in range(replay):
                noise = Variable(perturbation[0:x.size(0)], requires_grad=True).to(device)
                input = x + noise
                input.clamp(0, 1.0)

                y_ = model(input)
                loss = loss_calc(y_, y)

                optimizer.zero_grad()
                loss.backward()
                data_grad = noise.grad.data

                perturbation[0:x.size(0)] += koef_it * data_grad.sign()
                perturbation.clamp(min=-eps, max=eps)

                optimizer.step()

                temp_loss += loss.item()
                _, y_ = y_.max(1)
                temp_correct += y_.eq(y).sum().item()

                if (j == replay - 1):
                    train_total += y.size(0)
                    total_train_loss += temp_loss / replay
                    train_correct += temp_correct / replay

        total_train_acc = 100 * train_correct/train_total

        print(f"Total train loss for epoch {epoch+1}: {total_train_loss}")
        print(f"Total train accuracy for epoch {epoch+1}: {total_train_acc}")

        test_loss, test_acc = test(epoch)

        curr_epoch = f"epoch{epoch+1}"
        curr_dict = dict()
        curr_dict.update({"train_loss": total_train_loss, 
                          "train_accuracy": total_train_acc,
                          "test_loss": test_loss,
                          "test_accuracy": test_acc})
        
        train_stats.update({curr_epoch: curr_dict})

    total_time = time.time() - start_time
    train_stats.update({"train_time": total_time})

    file_path = f"./stats/{name}/stats.json"

    if (not os.path.exists(f"./stats/{name}")):
        os.mkdir(f"./stats/{name}")

    with open(file_path, "w") as file:
        json.dump(train_stats, file)

def test(curr_epoch=0):
    """
    Test function for the model initialized in the main function
    Params:
        curr_epoch: number of the current epoch (used for output)
    """
    model.eval()

    total_test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for (x, y) in test_loader:
            x = x.to(device)
            y = y.to(device)

            y_ = model(x)
            loss = loss_calc(y_, y)

            total_test_loss += loss.item()
            _, y_ = y_.max(1)
            test_total += y.size(0)
            test_correct += y_.eq(y).sum().item()

    total_test_acc = 100 * test_correct/test_total

    print(f"Total test loss for epoch {curr_epoch+1}: {total_test_loss}")
    print(f"Total test accuracy for epoch {curr_epoch+1}: {total_test_acc}")

    return (total_test_loss, total_test_acc)

def test_robustness():
    """
    Function for testing the robustness of the model initialized in the main function using adversarial images generated using PGD attack
    """
    model.eval()

    adv_total = 0
    adv_correct = 0

    for (x, y) in test_loader:
        x = x.to(device)
        y = y.to(device)
        adversarial = attack_pgd(model, x, y, eps=8/255, koef_it=1/255, steps=5, device=device)

        y_ = model(adversarial)
        _, y_ = y_.max(1)
        adv_total += y.size(0)
        adv_correct += y_.eq(y).sum().item()

    total_adv_acc = 100 * adv_correct/adv_total

    print(f"Accuracy on adversarial examples generated using PGD attack: {total_adv_acc}%")

    return total_adv_acc

if __name__ == "__main__":

    #print(f"Current device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

    classes_map = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    ###################################################################################################

    ##################################################
    # Train model and save it

    model = ResidualNetwork18().to(device)
    model_name = "resnet18_first"
    model_save_path = f"./models/{model_name}.pt"
    
    loss_calc = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)

    train(16, model_name)
    torch.save(model.state_dict(), model_save_path)

    ##################################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = "resnet18_first"
    # model_save_path = f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.3, weight_decay=5e-4)

    # test()
    # test_robustness()

    show_loss(model_name, save=True, show=False)
    show_accuracies(model_name, save=True, show=False)
    get_train_time(model_name)

    ####################################################################################################

    ##################################################

    # Train model using free adversarial training and save it

    # model = ResidualNetwork18().to(device)
    # model_name = "resnet18_first_free"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.3, weight_decay=5e-4)

    # train_free(16, model_name)
    # torch.save(model.state_dict(), model_save_path)

    # ##################################################
    # # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = "resnet18_first_free"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.3, weight_decay=5e-4)

    # test()
    # test_robustness()

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # get_train_time(model_name)