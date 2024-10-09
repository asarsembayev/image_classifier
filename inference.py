import yaml

import torch
from torchvision import transforms
from PIL import Image
import argparse

from models.cnn_custom_1 import Net


# Define the image transformations
transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def predict(model, image_path):
    # Load the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()


def main(args):
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    classes = {
        0: 'plane',
        1: 'car',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    
    # Load the model
    model = Net()
    model.load_state_dict(torch.load(config['paths']['output_dir'] + '/cifar_net.pth'))
    model.eval()
    prediction = predict(model, args.image_path)
    print(f"Predicted class: {classes[prediction]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image classification inference')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--config_path', type=str, default='config/config.yaml', help='Input path to a config file')

    args = parser.parse_args()
    
    main(args)
