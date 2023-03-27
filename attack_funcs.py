import torch
import torch.nn as nn

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