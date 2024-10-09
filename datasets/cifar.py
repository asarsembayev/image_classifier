import torch
import torchvision
import torchvision.transforms as transforms


class CIFAR10Dataset:
    def __init__(self, root='./data', train=True, transform=None, download=True):
        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, transform=self.transform, download=download
        )

    def get_loader(self, batch_size=4, shuffle=True, num_workers=2):
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )