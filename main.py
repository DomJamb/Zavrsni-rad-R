import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from ResidualNetwork18 import ResidualNetwork18

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())

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

    model = ResidualNetwork18().to(device)

    loss_calc = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(200):
        print(f"Starting epoch: {epoch}")

        #Train 
        model.train()

        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for (x, y) in train_loader:
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

        print(f"Total train loss for epoch{epoch}: {total_train_loss}")
        print(f"Train train accuracy for epoch{epoch}: {100 * train_correct/train_total}")

        #Test
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

        print(f"Total test loss for epoch{epoch}: {total_test_loss}")
        print(f"Train test accuracy for epoch{epoch}: {100 * test_correct/test_total}")

    torch.save(model, './models/resnet18.txt')