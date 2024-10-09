import os
import argparse

import yaml
import structlog
from tqdm import tqdm

from datasets import cifar
from models import cnn_custom_1

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision
import torch.optim as optim
import torchsummary


# Configure logger
logger = structlog.get_logger()

def configure_logging():
    structlog.configure(
        processors=[
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

configure_logging()


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on: {device}")

    # Load config file
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    input_dir = config['paths']['input_dir']
    output_dir = config['paths']['output_dir']
    logger.info(f"Input directory: {input_dir} | Output directory: {output_dir}")

    # Load dataset
    trainset = cifar.CIFAR10Dataset(root=input_dir, train=True, download=True)
    classes = trainset.classes
    trainloader = trainset.get_loader(batch_size=config['training']['batch_size'])

    testset = cifar.CIFAR10Dataset(root=input_dir, train=False, download=True)
    testloader = testset.get_loader(batch_size=config['training']['batch_size'])

    net = cnn_custom_1.Net().to(device)
    logger.info(torchsummary.summary(net, (3, 32, 32)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs) # f(x) -> net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                logger.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        logger.info('Finished Training')
        PATH = os.path.join(config['paths']['output_dir'], 'cifar_net.pth')
        os.makedirs(os.path.dirname(config['paths']['output_dir']), exist_ok=True)
        torch.save(net.state_dict(), PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-c', '--config_path', type=str, default='config/config.yaml', help='Input path to a config file')

    args = parser.parse_args()

    main(args)
