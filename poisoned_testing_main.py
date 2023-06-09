import os
import copy
import time
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from poisoned_testing.src.BadNets import BadNets
from norms import normalize_by_pnorm, clamp_by_pnorm

from ResidualNetwork18 import ResidualNetwork18
from util import get_train_time
from attack_funcs import attack_pgd, attack_pgd_l2
from graphing_funcs import show_accuracies, show_adversarial_accuracies, show_adversarial_accuracies_varying_steps, show_loss, show_train_accs, show_train_loss, compare_train_loss, compare_train_accs, compare_stats, show_poisoned_table, graph_poisoned_examples, graph_adv_examples_multiple_models, compare_change_of_predictions, compare_change_of_predictions_ratio, compare_change_of_predictions_difference
from AdvExampleVerbose import AdvExampleVerbose
from AdvExample import AdvExample

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_mixed(num_of_epochs, name, mixed_prec=True, train_loader=None, test_loader=None):
    """
    Train function (using mixed precision arithmetic) for the model initialized in the main function
    Params:
        num_of_epochs: total number of train epochs
        name: desired name for the model (used for saving the model parameters in a JSON file)
        mixed_prec: toggle for mixed precision training
        train_loader: train_loader
        test_loader: test_loader
    """
    start_time = time.time()
    train_stats = dict()

    train_losses = list()
    train_accuracies = list()

    if mixed_prec:
        scaler = GradScaler()

    for epoch in range(num_of_epochs):
        print(f"Starting epoch: {epoch + 1}")

        model.train()

        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            with autocast(enabled=mixed_prec):
                y_ = model(x)
                if (torch.any(torch.isnan(y_))):
                    print("NaNs in output detected.")
                loss = loss_calc(y_, y)

            optimizer.zero_grad()

            if mixed_prec:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()            
            else:
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()
            _, y_ = y_.max(1)
            train_total += y.size(0)
            train_correct += y_.eq(y).sum().item()

            train_losses.append(loss.item())
            train_accuracies.append(100 * (y_.eq(y).sum().item() / y.size(0)))

        total_train_acc = 100 * train_correct/train_total

        print(f"Total train loss for epoch {epoch+1}: {total_train_loss}")
        print(f"Total train accuracy for epoch {epoch+1}: {total_train_acc}")

        test_loss, test_acc = test(epoch, test_loader=test_loader)

        scheduler.step()

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
    loss_fp = f"./stats/{name}/train_loss.json"
    accs_fp = f"./stats/{name}/train_accs.json"

    if (not os.path.exists(f"./stats/{name}")):
        os.mkdir(f"./stats/{name}")

    with open(file_path, "w") as file:
        json.dump(train_stats, file)

    with open(loss_fp, "w") as file:
        json.dump(train_losses, file)

    with open(accs_fp, "w") as file:
        json.dump(train_accuracies, file)

def train_pgd(num_of_epochs, name, eps=8/255, alpha=1/255, steps=7, mixed_prec=True, train_loader=None, test_loader=None, poisoned_test_loader=None, limit=None):
    """
    Train function for the model initialized in the main function (implements training with PGD)
    Params:
        num_of_epochs: total number of train epochs
        name: desired name for the model (used for saving the model parameters in a JSON file)
        eps: maximum total perturbation
        alpha: maximum change threshold of individual pixels in given data for each iteration
        steps: number of iterations
        mixed_prec: toggle for mixed precision training
        train_loader: train loader
        test_loader: test loader
        poisoned_test_loader: test loader with poisoned data
        limit: Linf, L2, L1 (default Linf)
    """
    start_time = time.time()
    train_stats = dict()

    train_losses = list()
    train_accuracies = list()

    if mixed_prec:
        scaler = GradScaler()

    for epoch in range(num_of_epochs):
        print(f"Starting epoch: {epoch + 1}")

        model.train()

        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            if (i == 0):
                adv_x = x.clone().to(device)
                adv_y = y.clone().to(device)

            delta = torch.zeros_like(x)

            for _ in range(steps):
                delta = delta.to(device)
                delta.requires_grad = True

                with autocast(enabled=mixed_prec):
                    y_ = model(x + delta)
                    if (torch.any(torch.isnan(y_))):
                        print("NaNs in output detected.")
                    loss = loss_calc(y_, y)

                optimizer.zero_grad()

                if mixed_prec:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (limit == "l2"):
                    grad = delta.grad.data
                    grad = normalize_by_pnorm(grad)
                    delta = delta + alpha * grad
                    delta = torch.clamp(x + delta, min=0, max=1) - x
                    delta = clamp_by_pnorm(delta, 2, eps)
                else:
                    grad_sign = delta.grad.data.sign()
                    delta = delta + alpha * grad_sign
                    delta = torch.clamp(delta, min=-eps, max=eps)
                    delta = torch.clamp(x + delta, min=0, max=1) - x

                delta = delta.detach()

            adv_imgs = torch.clamp(x + delta, min=0, max=1)

            with autocast(enabled=mixed_prec):
                y_ = model(adv_imgs)
                if (torch.any(torch.isnan(y_))):
                    print("NaNs in output detected.")
                loss = loss_calc(y_, y)

            optimizer.zero_grad()

            if mixed_prec:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()
            _, y_ = y_.max(1)
            train_total += y.size(0)
            train_correct += y_.eq(y).sum().item()

            train_losses.append(loss.item())
            train_accuracies.append(100 * (y_.eq(y).sum().item() / y.size(0)))

        total_train_acc = 100 * train_correct/train_total

        print(f"Total train loss for epoch {epoch+1}: {total_train_loss}")
        print(f"Total train accuracy for epoch {epoch+1}: {total_train_acc}")

        test_loss, test_acc = test(epoch, test_loader=test_loader, name="normal")
        _, poisoned_test_acc = test(epoch, test_loader=poisoned_test_loader, name="poisoned")

        scheduler.step()

        adv_x = attack_pgd(model, adv_x, adv_y, eps=8/255, koef_it=1/255, steps=5, device=device).to(device) 
        adv_y_ = model(adv_x)
            
        _, adv_y_ = adv_y_.max(1)
        adv_total = adv_y.size(0)
        adv_correct = adv_y_.eq(adv_y).sum().item()
        adv_accuracy = 100 * adv_correct / adv_total

        curr_epoch = f"epoch{epoch+1}"
        curr_dict = dict()

        curr_dict.update({"train_loss": total_train_loss, 
                        "train_accuracy": total_train_acc,
                        "test_loss": test_loss,
                        "test_accuracy": test_acc,
                        "test_poisoned_accuracy": poisoned_test_acc,
                        "adv_accuracy": adv_accuracy})
        
        train_stats.update({curr_epoch: curr_dict})

    total_time = time.time() - start_time
    train_stats.update({"train_time": total_time})

    file_path = f"./stats/{name}/stats.json"
    loss_fp = f"./stats/{name}/train_loss.json"
    accs_fp = f"./stats/{name}/train_accs.json"

    if (not os.path.exists(f"./stats/{name}")):
        os.mkdir(f"./stats/{name}")

    with open(file_path, "w") as file:
        json.dump(train_stats, file)

    with open(loss_fp, "w") as file:
        json.dump(train_losses, file)

    with open(accs_fp, "w") as file:
        json.dump(train_accuracies, file)

def train_fast(num_of_epochs, name, eps=8/255, alpha=10/255, mixed_prec=True, early_stop=False, train_loader=None, test_loader=None, poisoned_test_loader=None):
    """
    Train function for the model initialized in the main function (implements Fast Adversarial Training)
    Params:
        num_of_epochs: total number of train epochs
        name: desired name for the model (used for saving the model parameters in a JSON file)
        eps: maximum total perturbation
        alpha: perturbation koefficient
        mixed_prec: toggle for mixed precision training
        early_stop: toggle for early stop evaluation
        train_loader: train loader
        test_loader: test loader
        poisoned_test_loader: test loader with poisoned data
    """
    start_time = time.time()
    train_stats = dict()

    train_losses = list()
    train_accuracies = list()

    prev_acc = 0

    if mixed_prec:
        scaler = GradScaler()

    for epoch in range(num_of_epochs):
        print(f"Starting epoch: {epoch + 1}")

        model.train()

        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            if (i == 0):
                adv_x = x
                adv_y = y

            noise = torch.zeros_like(x).uniform_(-eps, eps).to(device)
            with torch.no_grad():
                noise.add_(x).clamp_(0, 1).sub_(x) 
            noise.requires_grad = True

            input = x + noise

            with autocast(enabled=mixed_prec):
                y_ = model(input)
                if (torch.any(torch.isnan(y_))):
                    print("NaNs in output detected.")
                loss = loss_calc(y_, y)

            optimizer.zero_grad()

            if mixed_prec:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            data_grad = noise.grad.data

            noise = noise + alpha * data_grad.sign()
            with torch.no_grad():
                noise.clamp_(min=-eps, max=eps)
                noise.add_(x).clamp_(0, 1).sub_(x)

            noise = noise.detach()
            input = x + noise

            with autocast(enabled=mixed_prec):
                y_ = model(input)
                if (torch.any(torch.isnan(y_))):
                    print("NaNs in output detected.")
                loss = loss_calc(y_, y)

            optimizer.zero_grad()

            if mixed_prec:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()
            _, y_ = y_.max(1)
            train_total += y.size(0)
            train_correct += y_.eq(y).sum().item()

            train_losses.append(loss.item())
            train_accuracies.append(100 * (y_.eq(y).sum().item() / y.size(0)))

            scheduler.step()

        total_train_acc = 100 * train_correct/train_total

        print(f"Total train loss for epoch {epoch+1}: {total_train_loss}")
        print(f"Total train accuracy for epoch {epoch+1}: {total_train_acc}")

        test_loss, test_acc = test(epoch, test_loader=test_loader, name="normal")
        _, poisoned_test_acc = test(epoch, test_loader=poisoned_test_loader, name="poisoned")

        adv_x = attack_pgd(model, adv_x, adv_y, eps=8/255, koef_it=1/255, steps=5, device=device).to(device) 
        adv_y_ = model(adv_x)
            
        _, adv_y_ = adv_y_.max(1)
        adv_total = adv_y.size(0)
        adv_correct = adv_y_.eq(adv_y).sum().item()
        adv_accuracy = 100 * adv_correct / adv_total

        curr_epoch = f"epoch{epoch+1}"
        curr_dict = dict()

        if early_stop and (adv_accuracy < prev_acc - 20):
            train_losses = train_losses[:len(train_losses) - len(train_loader)]
            train_accuracies = train_accuracies[:len(train_accuracies) - len(train_loader)]
            break

        curr_dict.update({"train_loss": total_train_loss, 
                        "train_accuracy": total_train_acc,
                        "test_loss": test_loss,
                        "test_accuracy": test_acc,
                        "test_poisoned_accuracy": poisoned_test_acc,
                        "adv_accuracy": adv_accuracy})
        
        train_stats.update({curr_epoch: curr_dict})

        prev_acc = adv_accuracy

        best_model_states = copy.deepcopy(model.state_dict())

    total_time = time.time() - start_time
    train_stats.update({"train_time": total_time})

    file_path = f"./stats/{name}/stats.json"
    loss_fp = f"./stats/{name}/train_loss.json"
    accs_fp = f"./stats/{name}/train_accs.json"

    if (not os.path.exists(f"./stats/{name}")):
        os.mkdir(f"./stats/{name}")

    with open(file_path, "w") as file:
        json.dump(train_stats, file)

    with open(loss_fp, "w") as file:
        json.dump(train_losses, file)

    with open(accs_fp, "w") as file:
        json.dump(train_accuracies, file)

    model.load_state_dict(best_model_states)

def test(curr_epoch=None, test_loader=None, name=""):
    """
    Test function for the model initialized in the main function
    Params:
        curr_epoch: number of the current epoch (used for output)
    """
    model.eval()

    total_test_loss = 0
    test_correct = 0
    test_total = 0

    if name:
        print(f"Testing with {name} dataset...")

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

    if curr_epoch:
        print(f"Total test loss for epoch {curr_epoch+1}: {total_test_loss}")
        print(f"Total test accuracy for epoch {curr_epoch+1}: {total_test_acc}")
    else:
        print(f"Total test loss: {total_test_loss}")
        print(f"Total test accuracy: {total_test_acc}")

    return (total_test_loss, total_test_acc)

def test_robustness(num_steps=20, test_loader=None):
    """
    Function for testing the robustness of the model initialized in the main function using adversarial images generated using PGD attack
    """
    model.eval()

    adv_total = 0
    adv_correct = 0

    for (x, y) in test_loader:
        x = x.to(device)
        y = y.to(device)
        adversarial = attack_pgd(model, x, y, eps=8/255, koef_it=1/255, steps=num_steps, device=device)

        y_ = model(adversarial)
        _, y_ = y_.max(1)
        adv_total += y.size(0)
        adv_correct += y_.eq(y).sum().item()

    total_adv_acc = 100 * adv_correct/adv_total

    print(f"Accuracy on adversarial examples generated using PGD attack with {num_steps} steps: {total_adv_acc}%")

    return total_adv_acc

def test_robustness_multiple_steps(max_steps=20, test_loader=None):
    """
    Function for testing the robustness of the model initialized in the main function using adversarial images generated using PGD attack with varying number of steps
    """
    adv_accs = dict()

    for i in range(5, max_steps+1, 5):
        adv_accs.update({i: test_robustness(i, test_loader)})

    return adv_accs

def generate_adversarial_multiple_norms():
    """
    Function for generating adversarial examples for models using different norms
    """
    sampled_imgs = dict()
    adv_imgs = dict()

    imgs, _ = poisoned_test_loader.__iter__().__next__()
    _, labels = test_loader.__iter__().__next__()
    t = list()

    for i, label in enumerate(labels):
        l = label.item()
        if l not in sampled_imgs.keys():
            sampled_imgs.update({l: imgs[i]})
            t.append(AdvExample(l, np.array(imgs[i].cpu())))
        if len(sampled_imgs.keys()) == 3:
            break

    adv_imgs.update({"Zatrovane slike": t})
        
    model = ResidualNetwork18().to(device)
    model_name = f"resnet18_poisoned_pgd_l2_epochs_60_lr_0.1_eps512_255"
    model_save_path= f"./models/{model_name}.pt"
    model.load_state_dict(torch.load(model_save_path))

    eps=16/255
    alpha=1/255

    adv_x = attack_pgd(model, torch.stack(list(sampled_imgs.values())), torch.tensor([1,1,1]), eps=eps, koef_it=alpha, steps=20, device=device).to(device)
    y_ = model(adv_x)
    _, y_ = y_.max(1)

    t = list()
    for i, y in enumerate(y_):
        t.append(AdvExample(y.item(), np.array(adv_x[i].cpu())))
    
    adv_imgs.update({"Norma Linf": t})

    eps=512/255
    alpha=64/255

    adv_x = attack_pgd_l2(model, torch.stack(list(sampled_imgs.values())), torch.tensor([1,1,1]), eps=eps, alpha=alpha, steps=20, device=device).to(device)
    y_ = model(adv_x)
    _, y_ = y_.max(1)

    t = list()
    for i, y in enumerate(y_):
        t.append(AdvExample(y.item(), np.array(adv_x[i].cpu())))
    
    adv_imgs.update({"Norma L2": t})

    graph_adv_examples_multiple_models(adv_imgs, classes_map, "linf_l2", save=True, show=False)

def test_change_of_predictions():
    """
    Function for testing the number of changed predictions (comparison between normal and adversarial images) for a model (tested both on normal and poisoned dataset)
    """
    # eps = [2, 4, 6, 8, 10, 12, 14, 16]
    eps = [4, 8, 12, 16, 20, 24, 28, 32]
    stats = dict()
    stats.update({"eps": eps, "natural": list(), "poisoned": list()})

    model = ResidualNetwork18().to(device)
    model_name = f"resnet18_poisoned_pgd_epochs_60_lr_0.1"
    model_save_path= f"./models/{model_name}.pt"
    model.load_state_dict(torch.load(model_save_path))

    num_steps = 20
    alpha = 1/255

    for c in eps:
        print(f"Testing for epsilon {c}/255...")

        curr_eps = c / 255
        curr_natural = 0
        curr_poisoned = 0
        
        for (x, y) in test_loader:
            x = x.to(device)
            y = y.to(device)
            adversarial = attack_pgd(model, x, y, eps=curr_eps, koef_it=alpha, steps=num_steps, device=device)

            y_adv_ = model(adversarial)
            _, y_adv_ = y_adv_.max(1)

            y_ = model(x)
            _, y_ = y_.max(1)

            curr_natural += (y_adv_ != y_).sum()

        for (x, y) in poisoned_test_loader:
            x = x.to(device)
            y = y.to(device)
            adversarial = attack_pgd(model, x, y, eps=curr_eps, koef_it=alpha, steps=num_steps, device=device)

            y_adv_ = model(adversarial)
            _, y_adv_ = y_adv_.max(1)

            y_ = model(x)
            _, y_ = y_.max(1)

            curr_poisoned += (y_adv_ != y_).sum()

        stats["natural"].append(curr_natural.item())
        stats["poisoned"].append(curr_poisoned.item())

        print(f"Finished testing for epsilon {c}/255...\nTotal changed for natural dataset: {curr_natural}/10000 \nTotal changed for poisoned dataset: {curr_poisoned}/10000")

    file_path = f"./poisoned_stats/change_of_predictions_double.json"

    with open(file_path, "w") as file:
        json.dump(stats, file)

if __name__ == "__main__":

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

    if (not os.path.exists(f"./models")):
        os.mkdir(f"./models")
        print("Models dir created.")

    if (not os.path.exists(f"./stats")):
        os.mkdir(f"./stats")
        print("Stats dir created.")

    if (not os.path.exists(f"./poisoned_stats")):
        os.mkdir(f"./poisoned_stats")
        print("Poisoned stats dir created.")

    # Transforms and fetch of dataset

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Normal data loaders for train and evaluation

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

    # Initialization of badnet object for training

    pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
    pattern[0, -3:, -3:] = 255
    weight = torch.zeros((1, 32, 32), dtype=torch.float32)
    weight[0, -3:, -3:] = 1.0

    badnets_train = BadNets(
        train_dataset=train_data,
        test_dataset=test_data,
        model=ResidualNetwork18(),
        loss=nn.CrossEntropyLoss(),
        y_target=1,
        poisoned_rate=0.3,
        pattern=pattern,
        weight=weight,
        poisoned_target_transform_index=0,
        schedule=None,
        seed=666
    )

    badnets_test = BadNets(
        train_dataset=train_data,
        test_dataset=test_data,
        model=ResidualNetwork18(),
        loss=nn.CrossEntropyLoss(),
        y_target=1,
        poisoned_rate=1,
        pattern=pattern,
        weight=weight,
        poisoned_target_transform_index=0,
        schedule=None,
        seed=666
    )

    # Generating poisoned dataset

    poisoned_train_data, _ = badnets_train.get_poisoned_dataset()
    _, poisoned_test_data = badnets_test.get_poisoned_dataset()

    poisoned_train_loader = torch.utils.data.DataLoader(poisoned_train_data, batch_size=256, shuffle=True)
    poisoned_test_loader = torch.utils.data.DataLoader(poisoned_test_data, batch_size=100, shuffle=False)

    epochs = 60
    lr = 0.02

    ########################################
    # ResNet18 Natural, not poisoned
    ########################################

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_natural_not_poisoned_epochs_{epochs}_lr_{lr}"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train_mixed(epochs, model_name, train_loader=train_loader, test_loader=test_loader)
    # torch.save(model.state_dict(), model_save_path)

    ########################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_natural_not_poisoned_epochs_{epochs}_lr_{lr}"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 Natural, not poisoned")

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    # test(test_loader=test_loader, name="normal")
    # test(test_loader=poisoned_test_loader, name="poisoned")

    ########################################
    # ResNet18 Natural, poisoned
    ########################################

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_natural_poisoned_epochs_{epochs}_lr_{lr}"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train_mixed(epochs, model_name, train_loader=poisoned_train_loader, test_loader=test_loader)
    # torch.save(model.state_dict(), model_save_path)

    ########################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_natural_poisoned_epochs_{epochs}_lr_{lr}"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 Natural, poisoned")

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    # test(test_loader=test_loader, name="normal")
    # test(test_loader=poisoned_test_loader, name="poisoned")

    lr = 0.2

    ########################################
    # ResNet18 Fast, not poisoned
    ########################################

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_not_poisoned_fast_epochs_{epochs}_lr_{lr}"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # total_steps = epochs * len(train_loader)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=lr, step_size_up=(total_steps / 2), step_size_down=(total_steps / 2))

    # train_fast(epochs, model_name, train_loader=train_loader, test_loader=test_loader)
    # torch.save(model.state_dict(), model_save_path)

    ########################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_not_poisoned_fast_epochs_{epochs}_lr_{lr}"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 Fast, not poisoned")

    # robustness_over_steps = test_robustness_multiple_steps(test_loader=test_loader)

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies_varying_steps(robustness_over_steps, model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    # test(test_loader=test_loader, name="normal")
    # test(test_loader=poisoned_test_loader, name="poisoned")

    ########################################
    # ResNet18 Fast, poisoned
    ########################################

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_poisoned_fast_epochs_{epochs}_lr_{lr}"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # total_steps = epochs * len(train_loader)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=lr, step_size_up=(total_steps / 2), step_size_down=(total_steps / 2))

    # train_fast(epochs, model_name, train_loader=poisoned_train_loader, test_loader=test_loader, poisoned_test_loader=poisoned_test_loader)
    # torch.save(model.state_dict(), model_save_path)

    ########################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_poisoned_fast_epochs_{epochs}_lr_{lr}"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 Fast, poisoned")

    # robustness_over_steps = test_robustness_multiple_steps(test_loader=test_loader)

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies_varying_steps(robustness_over_steps, model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    # test(test_loader=test_loader, name="normal")
    # test(test_loader=poisoned_test_loader, name="poisoned")

    # compare_stats(model_name, "Resnet18 Fast, Poisoned", save=True, show=False)

    # models = {
    #     "resnet18_natural_not_poisoned_epochs_60_lr_0.02": "Natural training, not poisoned",
    #     "resnet18_natural_poisoned_epochs_60_lr_0.02": "Natural training, poisoned",
    #     "resnet18_not_poisoned_fast_epochs_60_lr_0.2": "Fast adversarial training, not poisoned",
    #     "resnet18_poisoned_fast_epochs_60_lr_0.2": "Fast adversarial training, poisoned",
    #     "resnet18_not_poisoned_pgd_epochs_60_lr_0.1": "PGD training, not poisoned",
    #     "resnet18_poisoned_pgd_epochs_60_lr_0.1": "PGD training, poisoned",
    #     "resnet18_not_poisoned_pgd_l2_epochs_60_lr_0.1_eps64_255": "PGD training L2 norm, not poisoned",
    #     "resnet18_poisoned_pgd_l2_epochs_60_lr_0.1_eps64_255": "PGD training L2 norm, poisoned",
    # }

    # models = {
    #     "resnet18_poisoned_pgd_l2_epochs_60_lr_0.1_eps64_255": "PGD training L2 norm, poisoned, e = 64/255",
    #     "resnet18_poisoned_pgd_l2_epochs_60_lr_0.1_eps128_255": "PGD training L2 norm, poisoned, e = 128/255",
    #     "resnet18_poisoned_pgd_l2_epochs_60_lr_0.1_eps256_255": "PGD training L2 norm, poisoned, e = 256/255",
    #     "resnet18_poisoned_pgd_l2_epochs_60_lr_0.1_eps512_255": "PGD training L2 norm, poisoned, e = 512/255",
    # }

    # compare_train_loss(models, "train_loss_comparison_all", save=True, show=False)
    # compare_train_accs(models, "train_accuracy_comparison_all", save=True, show=False)

    lr = 0.1

    ########################################
    # ResNet18 PGD, not poisoned
    ########################################

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_not_poisoned_pgd_epochs_{epochs}_lr_{lr}"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train_pgd(epochs, model_name, train_loader=train_loader, test_loader=test_loader, poisoned_test_loader=poisoned_test_loader)
    # torch.save(model.state_dict(), model_save_path)

    ########################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_not_poisoned_pgd_epochs_{epochs}_lr_{lr}"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 PGD, not poisoned")

    # robustness_over_steps = test_robustness_multiple_steps(test_loader=test_loader)

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies_varying_steps(robustness_over_steps, model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    # test(test_loader=test_loader, name="normal")
    # test(test_loader=poisoned_test_loader, name="poisoned")

    # compare_stats(model_name, "Resnet18 PGD, Not Poisoned", save=True, show=False)

    ########################################
    # ResNet18 PGD, poisoned
    ########################################

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_poisoned_pgd_epochs_{epochs}_lr_{lr}"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train_pgd(epochs, model_name, train_loader=poisoned_train_loader, test_loader=test_loader, poisoned_test_loader=poisoned_test_loader)
    # torch.save(model.state_dict(), model_save_path)

    ########################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_poisoned_pgd_epochs_{epochs}_lr_{lr}"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 PGD, poisoned")

    # robustness_over_steps = test_robustness_multiple_steps(test_loader=test_loader)

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies_varying_steps(robustness_over_steps, model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    # test(test_loader=test_loader, name="normal")
    # test(test_loader=poisoned_test_loader, name="poisoned")

    # compare_stats(model_name, "Resnet18 PGD, Poisoned", save=True, show=False)

    eps = 64/255
    alpha = 8/255

    # for i in range(0,4):
    #     ########################################
    #     # ResNet18 PGD L2 norm, varying eps, poisoned
    #     ########################################

    #     model = ResidualNetwork18().to(device)
    #     model_name = f"resnet18_poisoned_pgd_l2_epochs_{epochs}_lr_{lr}_eps{int(eps*255)}_255"
    #     model_save_path= f"./models/{model_name}.pt"
        
    #     loss_calc = nn.CrossEntropyLoss()
    #     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    #     train_pgd(epochs, model_name, eps=eps, alpha=alpha, train_loader=poisoned_train_loader, test_loader=test_loader, poisoned_test_loader=poisoned_test_loader, limit="l2")
    #     torch.save(model.state_dict(), model_save_path)

    #     ########################################
    #     # Load model and evaluate it
        
    #     model = ResidualNetwork18().to(device)
    #     model_name = f"resnet18_poisoned_pgd_l2_epochs_{epochs}_lr_{lr}_eps{int(eps*255)}_255"
    #     model_save_path= f"./models/{model_name}.pt"
    #     model.load_state_dict(torch.load(model_save_path))

    #     loss_calc = nn.CrossEntropyLoss()

    #     print(f"Resnet18 PGD, eps: {int(eps*255)}/255, poisoned")

    #     robustness_over_steps = test_robustness_multiple_steps(test_loader=test_loader)

    #     show_loss(model_name, save=True, show=False)
    #     show_accuracies(model_name, save=True, show=False)
    #     show_adversarial_accuracies(model_name, save=True, show=False)
    #     show_adversarial_accuracies_varying_steps(robustness_over_steps, model_name, save=True, show=False)
    #     show_train_loss(model_name, save=True, show=False)
    #     show_train_accs(model_name, save=True, show=False)
    #     get_train_time(model_name)

    #     test(test_loader=test_loader, name="normal")
    #     test(test_loader=poisoned_test_loader, name="poisoned")

    #     compare_stats(model_name, f"Resnet18 PGD, eps: {int(eps*255)}/255, Poisoned", save=True, show=False)

    #     eps *= 2
    #     alpha *= 2

    ########################################
    # ResNet18 PGD L2 norm, varying eps, not poisoned
    ########################################

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_not_poisoned_pgd_l2_epochs_{epochs}_lr_{lr}_eps{int(eps*255)}_255"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train_pgd(epochs, model_name, eps=eps, alpha=alpha, train_loader=train_loader, test_loader=test_loader, poisoned_test_loader=poisoned_test_loader, limit="l2")
    # torch.save(model.state_dict(), model_save_path)

    ########################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_not_poisoned_pgd_l2_epochs_{epochs}_lr_{lr}_eps{int(eps*255)}_255"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print(f"Resnet18 PGD, eps: {int(eps*255)}/255, not poisoned")

    # robustness_over_steps = test_robustness_multiple_steps(test_loader=test_loader)

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies_varying_steps(robustness_over_steps, model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    # test(test_loader=test_loader, name="normal")
    # test(test_loader=poisoned_test_loader, name="poisoned")

    # compare_stats(model_name, f"Resnet18 PGD, eps: {int(eps*255)}/255, Not Poisoned", save=True, show=False)

    # show_poisoned_table("zavrad_poisoned_stats_3.log", "poisoned_comparison_table", save=True, show=False)
    
    ##############################################################################
    # ResNet18 PGD L2 norm, max eps, poisoned (generating adversarial examples)
    ##############################################################################

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_poisoned_pgd_l2_epochs_60_lr_0.1_eps512_255"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # x, y = next(iter(poisoned_test_loader))
    # _, y_true = next(iter(test_loader))

    # x = x.to(device)
    # y = y.to(device)

    # eps=32/255

    # adv_x = attack_pgd(model, x, y, eps=eps, koef_it=2/255, steps=20, device=device).to(device)

    # adv_y_ = model(adv_x)
    # _, adv_y_ = adv_y_.max(1)

    # y_ = model(x)
    # _, y_ = y_.max(1)

    # adv_dict = dict()
    # for i in range(len(y_true)):
    #     if(classes_map[y_true[i].item()] in list(adv_dict.keys())):
    #         continue
    #     adv_dict.update({classes_map[y_true[i].item()]: [AdvExampleVerbose(classes_map[y_[i].item()], classes_map[adv_y_[i].item()], x[i].cpu().numpy(), adv_x[i].cpu().numpy())]})
    #     if len(list(adv_dict.keys())) == 3:
    #         break
    
    # graph_poisoned_examples(adv_dict, f"poisoned_adv_examples_eps_{int(eps*255)}_255", save=True, show=False)

    ##############################################################################
    # ResNet18 PGD L2 norm, max eps, poisoned (generating adversarial examples with L2 norm)
    ##############################################################################
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_poisoned_pgd_l2_epochs_60_lr_0.1_eps512_255"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # x, y = next(iter(poisoned_test_loader))
    # _, y_true = next(iter(test_loader))

    # x = x.to(device)
    # y = y.to(device)

    # eps=512/255
    # alpha=64/255

    # adv_x = attack_pgd_l2(model, x, y, eps=eps, alpha=alpha, steps=20, device=device).to(device)

    # adv_y_ = model(adv_x)
    # _, adv_y_ = adv_y_.max(1)

    # y_ = model(x)
    # _, y_ = y_.max(1)

    # adv_dict = dict()
    # for i in range(len(y_true)):
    #     if(classes_map[y_true[i].item()] in list(adv_dict.keys())):
    #         continue
    #     adv_dict.update({classes_map[y_true[i].item()]: [AdvExampleVerbose(classes_map[y_[i].item()], classes_map[adv_y_[i].item()], x[i].cpu().numpy(), adv_x[i].cpu().numpy())]})
    #     if len(list(adv_dict.keys())) == 3:
    #         break
    
    # graph_poisoned_examples(adv_dict, f"poisoned_adv_examples_l2_eps_{int(eps*255)}_255", save=True, show=False)
    # generate_adversarial_multiple_norms()
    # test_change_of_predictions()
    compare_change_of_predictions(save=True, show=False)
    compare_change_of_predictions_ratio(save=True, show=False)
    compare_change_of_predictions_difference(save=True, show=False)