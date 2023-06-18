import torch
import torch.nn as nn

from norms import normalize_by_pnorm, clamp_by_pnorm

def attack_pgd(model, images, labels, eps=8/255, koef_it=1/255, steps=7, device='cpu'):
    """
    Function for generating adversarial examples using the PGD attack
    Params:
        model:  arbitrary deep model
        images: data
        labels: correct labels for given data
        eps: maximum change threshold of individual pixels in given data
        koef_it: maximum change threshold of individual pixels in given data for each iteration
        steps: number of iterations
        device: device used
    Returns:
        adv_examples: adversarial examples generated using the PGD attack
    """
    model.eval()

    loss_calc = nn.CrossEntropyLoss()

    images = images.to(device)
    labels = labels.to(device)
    adv_examples = images.clone().detach()

    # Modify the images in each iteration
    for _ in range(steps):
        adv_examples = adv_examples.to(device)
        adv_examples.requires_grad = True

        # Calculate loss for given data
        model.zero_grad()
        probs = model(adv_examples)
        loss = loss_calc(probs, labels)

        loss.backward(inputs=[adv_examples])

        # Calculate gradient for given data
        data_grad = adv_examples.grad.data

        # Modify adversarial examples and clamp them so each pixel is not changed more than eps and is between 0 and 1
        adv_examples = adv_examples.detach() + koef_it * data_grad.sign()
        delta = torch.clamp(adv_examples - images, min=-eps, max=eps)
        adv_examples = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_examples

def attack_pgd_l2(model, x, y, eps=64/255, alpha=8/255, steps=7, device='cpu'):
    """
    Function for generating adversarial examples using the PGD attack with L2 norm
    Params:
        model: arbitrary deep model
        x: data
        y: correct labels for given data
        eps: maximum change threshold of individual pixels in given data
        alpha: maximum change threshold of individual pixels in given data for each iteration
        steps: number of iterations
        device: device used
    Returns:
        adv_imgs: adversarial examples generated using the PGD attack with L2 norm
    """
    model.eval()

    loss_calc = nn.CrossEntropyLoss()

    delta = torch.zeros_like(x)

    # Modify the images in each iteration
    for _ in range(steps):
        delta = delta.to(device)
        delta.requires_grad = True

        # Calculate loss for given data
        y_ = model(x + delta)
        loss = loss_calc(y_, y)

        model.zero_grad()

        loss.backward()

        # Calculate gradient for given data and modify adversarial examples
        grad = delta.grad.data
        grad = normalize_by_pnorm(grad)
        delta = delta + alpha * grad
        delta = torch.clamp(x + delta, min=0, max=1) - x
        delta = clamp_by_pnorm(delta, 2, eps)

        delta = delta.detach()

    adv_imgs = torch.clamp(x + delta, min=0, max=1)

    return adv_imgs

def attack_pgd_directed(model, images, labels, eps=8/255, koef_it=1/255, steps=7, target_class=0, device='cpu'):
    """
    Function for generating adversarial examples using the PGD attack (directed variant)
    Params:
        model:  arbitrary deep model
        images: data
        labels: correct labels for given data
        eps: maximum change threshold of individual pixels in given data
        koef_it: maximum change threshold of individual pixels in given data for each iteration
        steps: number of iterations
        target_class: target class for loss minimization
        device: device used
    Returns:
        adv_examples: adversarial examples generated using the PGD attack
    """
    model.eval()

    loss_calc = nn.CrossEntropyLoss()

    adv_examples = images.clone().detach()
    target_labels = torch.full(labels.shape, target_class).to(device)

    # Modify the images in each iteration
    for _ in range(steps):
        adv_examples = adv_examples.to(device)
        adv_examples.requires_grad = True

        # Calculate loss for given data
        model.zero_grad()
        probs = model(adv_examples)
        loss = loss_calc(probs, target_labels)

        loss.backward(inputs=[adv_examples])

        # Calculate gradient for given data
        data_grad = adv_examples.grad.data

        # Modify adversarial examples and clamp them so each pixel is not changed more than eps and is between 0 and 1
        adv_examples = adv_examples.detach() - koef_it * data_grad.sign()
        delta = torch.clamp(adv_examples.sub_(images), min=-eps, max=eps)
        adv_examples = torch.clamp(images.add_(delta), min=0, max=1).detach()

    return adv_examples