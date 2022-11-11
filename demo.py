import torch
import torchvision
from PIL import Image

label = ['airplane', 'automobile', 'bird', 'cat', 'deer',
         'dog', 'frog', 'horse', 'ship', 'truck']

image_path = "examples/airplane.png"
image = Image.open(image_path)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print("your image shape is", image.shape)

model = torch.load("./src/sz_75.pth", map_location=torch.device('cpu'))
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
argmax = output.argmax(1).numpy()[0]
pre = label[argmax]
print("The classification of the image is", pre)
