import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def attack_pgd(model, images, labels, eps=0.3, koef_it=0.05, steps=7):
    """
    Function for generating adversarial examples using the PGD attack
    Params:
        model:  arbitrary deep model
        images: data
        labels: correct labels for given data
        eps: maximum change threshold of individual pixels in given data
        koef_it: maximum change threshold of individual pixels in given data for each iteration
        steps: number of iterations
    Returns:
        adv_examples: adversarial examples generated using the PGD attack
    """
    model.eval()

    loss_calc = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    images = images.to(device)
    labels = labels.to(device)
    adv_examples = images.clone().detach()

    # Modify the images in each iteration
    for _ in range(steps):
        adv_examples = adv_examples.to(device)
        adv_examples.requires_grad = True

        # Calculate loss for given data
        optimizer.zero_grad()
        probs = model(adv_examples)
        loss = loss_calc(probs, labels)

        loss.backward()

        # Calculate gradient for given data
        data_grad = adv_examples.grad.data

        # Modify adversarial examples and clamp them so each pixel is not changed more than eps and is between 0 and 1
        adv_examples = adv_examples.detach() + koef_it * data_grad.sign()
        delta = torch.clamp(adv_examples - images, min=-eps, max=eps)
        adv_examples = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_examples