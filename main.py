import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from ResidualNetwork18 import ResidualNetwork18

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True)
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

    for epoch in range(10):
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

        print(f"Total train loss for epoch {epoch+1}: {total_train_loss}")
        print(f"Total train accuracy for epoch {epoch+1}: {100 * train_correct/train_total}")

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

        print(f"Total test loss for epoch {epoch+1}: {total_test_loss}")
        print(f"Total test accuracy for epoch {epoch+1}: {100 * test_correct/test_total}")

    torch.save(model, './models/resnet18.txt')