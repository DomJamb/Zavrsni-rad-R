import os
import json
import time
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

from attack_funcs import attack_pgd, attack_pgd_directed
from ResidualNetwork18 import ResidualNetwork18
from graphing_funcs import show_loss, show_accuracies, graph_adv_examples, show_train_loss, show_train_accs, show_adversarial_accuracies, show_adversarial_accuracies_varying_steps, show_stats, graph_adv_examples_multiple_models
from AdvExample import AdvExample
from util import get_train_time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(num_of_epochs, name):
    """
    Train function for the model initialized in the main function
    Params:
        num_of_epochs: total number of train epochs
        name: desired name for the model (used for saving the model parameters in a JSON file)
    """
    start_time = time.time()
    train_stats = dict()

    train_losses = list()
    train_accuracies = list()

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

            train_losses.append(loss.item())
            train_accuracies.append(100 * (y_.eq(y).sum().item() / y.size(0)))

        total_train_acc = 100 * train_correct/train_total

        print(f"Total train loss for epoch {epoch+1}: {total_train_loss}")
        print(f"Total train accuracy for epoch {epoch+1}: {total_train_acc}")

        test_loss, test_acc = test(epoch)

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

def train_mixed(num_of_epochs, name, mixed_prec=True):
    """
    Train function (using mixed precision arithmetic) for the model initialized in the main function
    Params:
        num_of_epochs: total number of train epochs
        name: desired name for the model (used for saving the model parameters in a JSON file)
        mixed_prec: toggle for mixed precision training
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

        test_loss, test_acc = test(epoch)

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

def train_replay(num_of_epochs, name, replay=4):
    """
    Train function for the model initialized in the main function (replays same batch n times)
    Params:
        num_of_epochs: total number of train epochs
        name: desired name for the model (used for saving the model parameters in a JSON file)
        replay: number of replays for each batch during 1 epoch
    """
    start_time = time.time()
    train_stats = dict()

    train_losses = list()
    train_accuracies = list()

    num_of_epochs = math.ceil(num_of_epochs/replay)

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
                y_ = model(x)
                loss = loss_calc(y_, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                temp_loss += loss.item()
                _, y_ = y_.max(1)
                temp_correct += y_.eq(y).sum().item()

                if (j == replay - 1):
                    train_total += y.size(0)
                    total_train_loss += temp_loss / replay
                    train_correct += temp_correct / replay

                    train_losses.append(temp_loss / replay)
                    train_accuracies.append(100 * ((temp_correct / replay) / y.size(0)))

        total_train_acc = 100 * train_correct/train_total

        print(f"Total train loss for epoch {epoch+1}: {total_train_loss}")
        print(f"Total train accuracy for epoch {epoch+1}: {total_train_acc}")

        test_loss, test_acc = test(epoch)

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

def train_pgd(num_of_epochs, name, eps=8/255, alpha=2/255, steps=7, mixed_prec=True):
    """
    Train function for the model initialized in the main function (implements training with PGD)
    Params:
        num_of_epochs: total number of train epochs
        name: desired name for the model (used for saving the model parameters in a JSON file)
        eps: maximum total perturbation
        alpha: maximum change threshold of individual pixels in given data for each iteration
        steps: number of iterations
        mixed_prec: toggle for mixed precision training
    """
    start_time = time.time()
    train_stats = dict()

    adv_examples = dict()

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
                adv_x = x
                adv_y = y

            adv_imgs = x.clone().detach()
            adv_imgs = torch.clamp(adv_imgs + torch.zeros_like(adv_imgs).uniform_(-eps, eps), min=0, max=1)

            for _ in range(steps):
                adv_imgs = adv_imgs.to(device)
                adv_imgs.requires_grad = True

                with autocast(enabled=mixed_prec):
                    y_ = model(adv_imgs)
                    if (torch.any(torch.isnan(y_))):
                        print("NaNs in output detected.")
                    loss = loss_calc(y_, y)

                optimizer.zero_grad()

                if mixed_prec:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                data_grad = adv_imgs.grad.data

                with torch.no_grad():
                    adv_imgs = adv_imgs.detach() + alpha * data_grad.sign()
                    delta = torch.clamp(adv_imgs - x, min=-eps, max=eps)
                    adv_imgs = torch.clamp(x + delta, min=0, max=1).detach()

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
            
            if (((epoch + 1) % 16 == 0) and (i == 0)):
                adv_list = list()
                for i in range(4):
                    adv_example = AdvExample(classes_map[y[i].item()], (adv_imgs[i]).detach().cpu().numpy())
                    adv_list.append(adv_example)
                adv_examples.update({epoch+1: adv_list})

        total_train_acc = 100 * train_correct/train_total

        print(f"Total train loss for epoch {epoch+1}: {total_train_loss}")
        print(f"Total train accuracy for epoch {epoch+1}: {total_train_acc}")

        test_loss, test_acc = test(epoch)

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

    graph_adv_examples(adv_examples, name, save=True, show=False)

def train_free(num_of_epochs, name, replay=4, eps=8/255, koef_it=1/255, mixed_prec=True):
    """
    Train function for the model initialized in the main function (implements Free Adversarial Training)
    Params:
        num_of_epochs: total number of train epochs
        name: desired name for the model (used for saving the model parameters in a JSON file)
        replay: number of replays for each batch during 1 epoch
        eps: maximum total perturbation
        koef_it: increment perturbation 
        mixed_prec: toggle for mixed precision training
    """
    start_time = time.time()
    train_stats = dict()

    tensor_size = [train_loader.batch_size, 3, 32, 32]
    perturbation = torch.zeros(*tensor_size).to(device)

    num_of_epochs = math.ceil(num_of_epochs/replay)

    adv_examples = dict()

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
                adv_x = x
                adv_y = y

            if (((epoch + 1) % (16 / replay) == 0) and i == 0):
                adv_list = list()
                for j in range(4):
                    adv_example = AdvExample(classes_map[y[j].item()], (x[j] + perturbation[j]).clamp(0, 1.0).detach().cpu().numpy())
                    adv_list.append(adv_example)
                adv_examples.update({epoch+1: adv_list})

            temp_loss = 0
            temp_correct = 0

            for j in range(replay):
                noise = Variable(perturbation[0:x.size(0)], requires_grad=True).to(device)
                with torch.no_grad():
                    noise.add_(x).clamp_(0, 1).sub_(x)
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

                perturbation[0:x.size(0)] += koef_it * data_grad.sign()
                with torch.no_grad():
                    perturbation.clamp_(min=-eps, max=eps)

                if mixed_prec:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                temp_loss += loss.item()
                _, y_ = y_.max(1)
                temp_correct += y_.eq(y).sum().item()

                if (j == replay - 1):
                    train_total += y.size(0)
                    total_train_loss += temp_loss / replay
                    train_correct += temp_correct / replay

                    train_losses.append(temp_loss / replay)
                    train_accuracies.append(100 * ((temp_correct / replay) / y.size(0)))

        total_train_acc = 100 * train_correct/train_total

        print(f"Total train loss for epoch {epoch+1}: {total_train_loss}")
        print(f"Total train accuracy for epoch {epoch+1}: {total_train_acc}")

        test_loss, test_acc = test(epoch)

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

    graph_adv_examples(adv_examples, name, save=True, show=False)

def train_fast(num_of_epochs, name, eps=8/255, alpha=10/255, mixed_prec=True, early_stop=False):
    """
    Train function for the model initialized in the main function (implements Fast Adversarial Training)
    Params:
        num_of_epochs: total number of train epochs
        name: desired name for the model (used for saving the model parameters in a JSON file)
        eps: maximum total perturbation
        alpha: perturbation koefficient
        mixed_prec: toggle for mixed precision training
        early_stop: toggle for early stop evaluation
    """
    start_time = time.time()
    train_stats = dict()

    adv_examples = dict()

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
            
            if (((epoch + 1) % 16 == 0) and (i == 0)):
                adv_list = list()
                for i in range(4):
                    adv_example = AdvExample(classes_map[y[i].item()], (input[i]).detach().cpu().numpy())
                    adv_list.append(adv_example)
                adv_examples.update({epoch+1: adv_list})

        total_train_acc = 100 * train_correct/train_total

        print(f"Total train loss for epoch {epoch+1}: {total_train_loss}")
        print(f"Total train accuracy for epoch {epoch+1}: {total_train_acc}")

        test_loss, test_acc = test(epoch)

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

    graph_adv_examples(adv_examples, name, save=True, show=False)

def train_fast_plus(num_of_epochs, name, eps=8/255, alpha_fast=10/255, alpha_pgd=1/255, steps=10, mixed_prec=True, early_stop=False, pgd_start_epoch=0):
    """
    Train function for the model initialized in the main function (implements Fast+ Adversarial Training)
    Params:
        num_of_epochs: total number of train epochs
        name: desired name for the model (used for saving the model parameters in a JSON file)
        eps: maximum total perturbation
        alpha_fast: perturbation koefficient for fast training
        alpha_pgd: perturbation koefficient for pgd training
        steps: number of steps for PGD train
        mixed_prec: toggle for mixed precision training
        early_stop: toggle for early stop evaluation
        pgd_start_epoch: first desired epoch for pgd training
    """
    start_time = time.time()
    train_stats = dict()

    adv_examples = dict()

    train_losses = list()
    train_accuracies = list()

    prev_acc = 0
    last_batches_acc = 0
    use_fast = True

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

            if use_fast:
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

                noise = noise + alpha_fast * data_grad.sign()
                with torch.no_grad():
                    noise.clamp_(min=-eps, max=eps)
                    noise.add_(x).clamp_(0, 1).sub_(x)

                noise = noise.detach()
                input = x + noise
            else:
                input = x.clone().detach()
                input = torch.clamp(input + torch.zeros_like(input).uniform_(-eps, eps), min=0, max=1)

                for _ in range(steps):
                    input = input.to(device)
                    input.requires_grad = True

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

                    data_grad = input.grad.data

                    with torch.no_grad():
                        input = input.detach() + alpha_pgd * data_grad.sign()
                        delta = torch.clamp(input - x, min=-eps, max=eps)
                        input = torch.clamp(x + delta, min=0, max=1).detach()

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
            
            if (((epoch + 1) % 16 == 0) and (i == 0)):
                adv_list = list()
                for i in range(4):
                    adv_example = AdvExample(classes_map[y[i].item()], (input[i]).detach().cpu().numpy())
                    adv_list.append(adv_example)
                adv_examples.update({epoch+1: adv_list})

            if ((i + 1) % 20 == 0):
                if ((pgd_start_epoch and epoch + 1 < pgd_start_epoch) or not pgd_start_epoch):
                    use_fast = True

                    idx = torch.randint(0, len(test_loader), (1,)).item()
                    i = 0
                    for x_check, y_check in test_loader:
                        if i == idx:
                            break
                        i += 1

                    x_check = x_check.to(device)
                    y_check = y_check.to(device)

                    x_check = attack_pgd(model, x_check, y_check, eps=8/255, koef_it=1/255, steps=5, device=device).to(device) 
                    y_check_ = model(x_check)
                    model.train()

                    _, y_check_ = y_check_.max(1)
                    total = y_check.size(0)
                    correct = y_check_.eq(y_check).sum().item()

                    curr_acc = 100 * correct / total

                    if last_batches_acc > curr_acc + 10:
                        use_fast = False

                    last_batches_acc = curr_acc
                else:
                    use_fast = False

        total_train_acc = 100 * train_correct/train_total

        print(f"Total train loss for epoch {epoch+1}: {total_train_loss}")
        print(f"Total train accuracy for epoch {epoch+1}: {total_train_acc}")

        test_loss, test_acc = test(epoch)
        
        scheduler.step()

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
                        "adv_accuracy": adv_accuracy})
        
        train_stats.update({curr_epoch: curr_dict})

        prev_acc = adv_accuracy

        best_model_states = copy.deepcopy(model.state_dict())

        if (pgd_start_epoch and epoch + 2 == pgd_start_epoch):
            use_fast = False

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

    graph_adv_examples(adv_examples, name, save=True, show=False)

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

def test_robustness(num_steps=20):
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

def test_robustness_multiple_steps(max_steps=20):
    """
    Function for testing the robustness of the model initialized in the main function using adversarial images generated using PGD attack with varying number of steps
    """
    adv_accs = dict()

    for i in range(5, max_steps+1, 5):
        adv_accs.update({i: test_robustness(i)})

    return adv_accs

def sample_adv_examples_multiple_models():
    """
    Function for creating adversarial examples for multiple trained models
    """
    sampled_imgs = dict()
    adv_imgs = dict()

    imgs, labels = test_loader.__iter__().__next__()
    t = list()

    for i, label in enumerate(labels):
        l = label.item()
        if l not in sampled_imgs.keys():
            sampled_imgs.update({l: imgs[i]})
            t.append(AdvExample(l, np.array(imgs[i].cpu())))
        if len(sampled_imgs.keys()) == 6:
            break

    adv_imgs.update({"Prirodne slike": t})

    ##### PGD

    model = ResidualNetwork18().to(device)
    model_name = f"resnet18_pgd_epochs_80_lr_0.1"
    model_save_path= f"./models/{model_name}.pt"
    model.load_state_dict(torch.load(model_save_path))

    adv_x = attack_pgd(model, torch.stack(list(sampled_imgs.values())), torch.tensor(list(sampled_imgs.keys())), eps=8/255, koef_it=1/255, steps=20, device=device)
    y_ = model(adv_x)
    _, y_ = y_.max(1)

    t = list()
    for i, y in enumerate(y_):
        t.append(AdvExample(y.item(), np.array(adv_x[i].cpu())))
    
    adv_imgs.update({"Algoritam PGD": t})

    ##### FreeAdv

    model = ResidualNetwork18().to(device)
    model_name = f"resnet18_free_epochs_10_replay_8_lr_0.1"
    model_save_path= f"./models/{model_name}.pt"
    model.load_state_dict(torch.load(model_save_path))

    adv_x = attack_pgd(model, torch.stack(list(sampled_imgs.values())), torch.tensor(list(sampled_imgs.keys())), eps=8/255, koef_it=1/255, steps=20, device=device)
    y_ = model(adv_x)
    _, y_ = y_.max(1)

    t = list()
    for i, y in enumerate(y_):
        t.append(AdvExample(y.item(), np.array(adv_x[i].cpu())))
    
    adv_imgs.update({"Algoritam FreeAdv": t})

    ##### FastAdv, Early

    model = ResidualNetwork18().to(device)
    model_name = f"resnet18_fast_epochs_80_lr_0.2_early"
    model_save_path= f"./models/{model_name}.pt"
    model.load_state_dict(torch.load(model_save_path))

    adv_x = attack_pgd(model, torch.stack(list(sampled_imgs.values())), torch.tensor(list(sampled_imgs.keys())), eps=8/255, koef_it=1/255, steps=20, device=device)
    y_ = model(adv_x)
    _, y_ = y_.max(1)

    t = list()
    for i, y in enumerate(y_):
        t.append(AdvExample(y.item(), np.array(adv_x[i].cpu())))
    
    adv_imgs.update({"Algoritam FastAdv, Early": t})

    ##### FastAdv+, Early

    model = ResidualNetwork18().to(device)
    model_name = f"resnet18_fast+_epochs_80_lr_0.2_early"
    model_save_path= f"./models/{model_name}.pt"
    model.load_state_dict(torch.load(model_save_path))

    adv_x = attack_pgd(model, torch.stack(list(sampled_imgs.values())), torch.tensor(list(sampled_imgs.keys())), eps=8/255, koef_it=1/255, steps=20, device=device)
    y_ = model(adv_x)
    _, y_ = y_.max(1)

    t = list()
    for i, y in enumerate(y_):
        t.append(AdvExample(y.item(), np.array(adv_x[i].cpu())))
    
    adv_imgs.update({"Algoritam FastAdv+, Early": t})

    ##### FastAdvW, Early

    model = ResidualNetwork18().to(device)
    model_name = f"resnet18_fastw_epochs_80_lr_0.2_early"
    model_save_path= f"./models/{model_name}.pt"
    model.load_state_dict(torch.load(model_save_path))

    adv_x = attack_pgd(model, torch.stack(list(sampled_imgs.values())), torch.tensor(list(sampled_imgs.keys())), eps=8/255, koef_it=1/255, steps=20, device=device)
    y_ = model(adv_x)
    _, y_ = y_.max(1)

    t = list()
    for i, y in enumerate(y_):
        t.append(AdvExample(y.item(), np.array(adv_x[i].cpu())))
    
    adv_imgs.update({"Algoritam FastAdvW, Early": t})

    graph_adv_examples_multiple_models(adv_imgs, classes_map, "adv_imgs_multiple_models", save=True, show=False)

if __name__ == "__main__":

    print(f"Current device: {device}")

    if (not os.path.exists(f"./models")):
        os.mkdir(f"./models")
        print("Models dir created.")

    if (not os.path.exists(f"./stats")):
        os.mkdir(f"./stats")
        print("Stats dir created.")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

    epochs = 80
    replay = 8

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
    # ResNet18
    ##################################################
    # Train model and save it

    # model = ResidualNetwork18().to(device)
    # model_name = "resnet18_first"
    # model_save_path = f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train(epochs, model_name)
    # torch.save(model.state_dict(), model_save_path)

    ##################################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = "resnet18_first"
    # model_save_path = f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18")
    # test()
    # test_robustness()

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    ####################################################################################################
    # ResNet18 Mixed precision
    ##################################################

    # Train model using mixed precision training and save it

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_mixed_epochs_{epochs}_lr_0.02"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train_mixed(epochs, model_name)
    # torch.save(model.state_dict(), model_save_path)

    ##################################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_mixed_epochs_{epochs}_lr_0.02"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 Mixed Precision")
    # test()
    # test_robustness()

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    ####################################################################################################
    # ResNet18 Replay
    ##################################################

    # Train model using training with replay and save it

    # model = ResidualNetwork18().to(device)
    # model_name = "resnet18_first_replay"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(math.ceil(epochs/replay)))

    # train_replay(epochs, model_name, replay)
    # torch.save(model.state_dict(), model_save_path)

    # ##################################################
    # # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = "resnet18_first_replay"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 Replay")
    # test()
    # test_robustness()

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    ####################################################################################################
    # ResNet18 PGD
    ##################################################

    # Train model using PGD training and save it

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_pgd_epochs_{epochs}_lr_0.1"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train_pgd(epochs, model_name)
    # torch.save(model.state_dict(), model_save_path)

    ##################################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_pgd_epochs_{epochs}_lr_0.1"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 PGD")
    # test()
    # test_robustness()

    # robustness_over_steps = test_robustness_multiple_steps()

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies_varying_steps(robustness_over_steps, model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    ####################################################################################################
    # ResNet18 Free
    ##################################################

    # Train model using free adversarial training and save it

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_free_epochs_{math.ceil(epochs/replay)}_replay_{replay}_lr_0.1"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(math.ceil(epochs/replay)))

    # train_free(epochs, model_name, replay)
    # torch.save(model.state_dict(), model_save_path)

    # ##################################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_free_epochs_{math.ceil(epochs/replay)}_replay_{replay}_lr_0.1"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 Free")
    # test()
    # test_robustness()

    # robustness_over_steps = test_robustness_multiple_steps()

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies_varying_steps(robustness_over_steps, model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    ####################################################################################################
    # ResNet18 Fast without Early Stop
    ##################################################

    # Train model using fast adversarial training and save it

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_fast_epochs_{epochs}_lr_0.2_no_early"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)

    # total_steps = epochs * len(train_loader)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=0.2, step_size_up=(total_steps / 2), step_size_down=(total_steps / 2))

    # train_fast(epochs, model_name)
    # torch.save(model.state_dict(), model_save_path)

    ##################################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_fast_epochs_{epochs}_lr_0.2_no_early"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 Fast, no early stop")
    # test()
    # test_robustness()

    # robustness_over_steps = test_robustness_multiple_steps()

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies_varying_steps(robustness_over_steps, model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    ####################################################################################################
    # ResNet18 Fast with Early Stop
    ##################################################

    # Train model using fast adversarial training and save it

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_fast_epochs_{epochs}_lr_0.2_early"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)

    # total_steps = epochs * len(train_loader)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=0.2, step_size_up=(total_steps / 2), step_size_down=(total_steps / 2))

    # train_fast(epochs, model_name, early_stop=True)
    # torch.save(model.state_dict(), model_save_path)

    ##################################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_fast_epochs_{epochs}_lr_0.2_early"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 Fast, early stop")
    # test()
    # test_robustness()

    # robustness_over_steps = test_robustness_multiple_steps()

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies_varying_steps(robustness_over_steps, model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    ####################################################################################################
    # ResNet18 Fast+ without Early Stop
    ##################################################

    # Train model using fast+ adversarial training and save it

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_fast+_epochs_{epochs}_lr_0.2_no_early"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train_fast_plus(epochs, model_name)
    # torch.save(model.state_dict(), model_save_path)

    ##################################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_fast+_epochs_{epochs}_lr_0.2_no_early"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 Fast+, no early stop")
    # test()
    # test_robustness()

    # robustness_over_steps = test_robustness_multiple_steps()

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies_varying_steps(robustness_over_steps, model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    ####################################################################################################
    # ResNet18 Fast+ with Early Stop
    ##################################################

    # Train model using fast+ adversarial training and save it

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_fast+_epochs_{epochs}_lr_0.2_early"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train_fast_plus(epochs, model_name, early_stop=True)
    # torch.save(model.state_dict(), model_save_path)

    ##################################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_fast+_epochs_{epochs}_lr_0.2_early"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 Fast+, early stop")
    # test()
    # test_robustness()

    # robustness_over_steps = test_robustness_multiple_steps()

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies_varying_steps(robustness_over_steps, model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    ####################################################################################################
    # ResNet18 FastW without Early Stop
    ##################################################

    # Train model using fastW adversarial training and save it

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_fastw_epochs_{epochs}_lr_0.2_no_early"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train_fast_plus(epochs, model_name, pgd_start_epoch=71)
    # torch.save(model.state_dict(), model_save_path)

    ##################################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_fastw_epochs_{epochs}_lr_0.2_no_early"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 FastW, no early stop")
    # test()
    # test_robustness()

    # robustness_over_steps = test_robustness_multiple_steps()

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies_varying_steps(robustness_over_steps, model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    ####################################################################################################
    # ResNet18 FastW with Early Stop
    ##################################################

    # Train model using fastW adversarial training and save it

    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_fastw_epochs_{epochs}_lr_0.2_early"
    # model_save_path= f"./models/{model_name}.pt"
    
    # loss_calc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # train_fast_plus(epochs, model_name, early_stop=True, pgd_start_epoch=71)
    # torch.save(model.state_dict(), model_save_path)

    ##################################################
    # Load model and evaluate it
    
    # model = ResidualNetwork18().to(device)
    # model_name = f"resnet18_fastw_epochs_{epochs}_lr_0.2_early"
    # model_save_path= f"./models/{model_name}.pt"
    # model.load_state_dict(torch.load(model_save_path))

    # loss_calc = nn.CrossEntropyLoss()

    # print("Resnet18 FastW, early stop")
    # test()
    # test_robustness()

    # robustness_over_steps = test_robustness_multiple_steps()

    # show_loss(model_name, save=True, show=False)
    # show_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies(model_name, save=True, show=False)
    # show_adversarial_accuracies_varying_steps(robustness_over_steps, model_name, save=True, show=False)
    # show_train_loss(model_name, save=True, show=False)
    # show_train_accs(model_name, save=True, show=False)
    # get_train_time(model_name)

    # show_stats("all_models_80_epochs_stats_short.log", "stats_comparison", save=True, show=False)
    sample_adv_examples_multiple_models()