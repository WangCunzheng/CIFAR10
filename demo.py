import torch
import torchvision
from PIL import Image
import argparse


def main(args):

    label = ['airplane', 'automobile', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck']

    image_path = args.i
    image = Image.open(image_path)
    image = image.convert('RGB')
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                torchvision.transforms.ToTensor()])

    image = transform(image)
    print("your image shape is", image.shape)

    model = torch.load(args.m, map_location=torch.device('cpu'))
    image = torch.reshape(image, (1, 3, 32, 32))
    model.eval()
    with torch.no_grad():
        output = model(image)
    argmax = output.argmax(1).numpy()[0]
    pre = label[argmax]
    print("The classification of the image is", pre)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', type=str, default='./examples/airplane.png', help="Image path")
    parser.add_argument('-m', type=str, default='./src/sz_75.pth', help="Model path")
    args = parser.parse_args()
    main(args)
