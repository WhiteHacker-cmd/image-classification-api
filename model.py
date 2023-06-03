import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")




normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)


transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
])



# class VGG(nn.Module):
#     def __init__(self, n_classes):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(256, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2)
#         )
#         self.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(7*7*512, 4096),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(),
#             nn.Linear(4096,n_classes),
            
#         )
    
    
#     def forward(self, x):
#         out = self.conv(x)
#         out = out.view(out.size(0), -1)
        
#         out = self.fc(out)
#         return out



# model = torch.load("vgg16.pt")
# print(model.eval())


# model_scripted = torch.jit.script(model)# Export to TorchScript

# model_scripted.save('model_scripted.pt') # Save



model = torch.load("model_scripted.pt")

print(model.eval())


def predict(img_path):
    img = Image.open(img_path)
    img = transform(img)
    img = img.view(1, 3, 224, 224).to(device)



    # img = img.to(torch.device('cpu'))
    # img = img/2 + 0.5
    # npimage = img.numpy()
    # plt.imshow(np.transpose(npimage, (1,2,0)))
    # plt.show()




    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    return classes[predicted]


# predict("deer.jpg")