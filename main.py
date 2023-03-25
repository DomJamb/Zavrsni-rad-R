import json
import time
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from attack_funcs import attack_pgd
from ResidualNetwork18 import ResidualNetwork18
from graphing_funcs import show_loss, show_accuracies

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(num_of_epochs, name):
    start_time = time.time()
    train_stats = dict()
    for epoch in range(num_of_epochs):
        print(f"Starting epoch: {epoch + 1}")

        #Train 
        model.train()

        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_ = model(x)
            loss = loss_calc(y_, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss
            _, y_ = y_.max(1)
            train_total += y.size(0)
            train_correct += y_.eq(y).sum().item()

        total_train_acc = 100 * train_correct/train_total

        print(f"Total train loss for epoch {epoch+1}: {total_train_loss}")
        print(f"Total train accuracy for epoch {epoch+1}: {total_train_acc}")

        test_loss, test_acc = test(epoch)

        curr_epoch = f"epoch{epoch+1}"
        curr_dict = dict()
        curr_dict.update({"train_loss": total_train_loss.item(), 
                          "train_accuracy": total_train_acc,
                          "test_loss": test_loss.item(),
                          "test_accuracy": test_acc})
        
        train_stats.update({curr_epoch: curr_dict})

    total_time = time.time() - start_time
    train_stats.update({"train_time": total_time})

    path = f"./stats/{name}/stats.json"
    with open(path, "w") as file:
        json.dump(train_stats, file)

def test(curr_epoch=0):
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

            total_test_loss += loss
            _, y_ = y_.max(1)
            test_total += y.size(0)
            test_correct += y_.eq(y).sum().item()

    total_test_acc = 100 * test_correct/test_total

    print(f"Total test loss for epoch {curr_epoch+1}: {total_test_loss}")
    print(f"Total test accuracy for epoch {curr_epoch+1}: {total_test_acc}")

    return (total_test_loss, total_test_acc)

def test_robustness():
    model.eval()

    adv_total = 0
    adv_correct = 0

    for (x, y) in test_loader:
        x = x.to(device)
        y = y.to(device)
        adversarial = attack_pgd(model, x, y, eps=0.3, koef_it=0.05, steps=5, device=device)

        y_ = model(adversarial)
        _, y_ = y_.max(1)
        adv_total += y.size(0)
        adv_correct += y_.eq(y).sum().item()

    total_adv_acc = 100 * adv_correct/adv_total

    print(f"Accuracy on adversarial examples generated using PGD attack: {total_adv_acc}")

    return total_adv_acc

if __name__ == "__main__":

    #print(f"Current device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
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

    ##################################################
    # Train model and save it

    # model = ResidualNetwork18().to(device)
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.15, weight_decay=5e-4)

    # train(10, "resnet18_first")
    # torch.save(model.state_dict(), './models/resnet18_first.pt')

    ##################################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model.load_state_dict(torch.load('./models/resnet18_first.pt'))

    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.15, weight_decay=5e-4)

    # test_robustness()

    # show_loss('resnet18_first', save=True, show=False)
    # show_accuracies('resnet18_first', save=True, show=False)