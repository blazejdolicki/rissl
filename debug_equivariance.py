#!/usr/bin/env python
# coding: utf-8

# # General E(2)-Equivariant Steerable CNNs  -  A concrete example


# In[1]:


import torch

from e2cnn import gspaces
from e2cnn import nn

import sys
# sys.path.insert(0, "/home/b.dolicki/thesis/")
sys.path.insert(0, "D://Blazej//Dokumenty//AI MSc//Thesis//thesis")
from rissl.models.e2_resnet import  e2_resnet18

# Let's try the model on *rotated* MNIST

# download the dataset
# get_ipython().system('wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip')
# uncompress the zip file
# get_ipython().system('unzip -n mnist_rotation_new.zip -d mnist_rotation_new')

from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose

import numpy as np

from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model_type', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--N', type=int)
args = parser.parse_args()

# Build the dataset

class MnistRotDataset(Dataset):
    
    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']
            
        if mode == "train":
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"
        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)

# images are padded to have shape 29x29.
# this allows to use odd-size filters with stride 2 when downsampling a feature map in the model
pad = Pad((0, 0, 1, 1), fill=0)

# to reduce interpolation artifacts (e.g. when testing the model on rotated images),
# we upsample an image by a factor of 3, rotate it and finally downsample it again
# resize1 = Resize(87)
# resize2 = Resize(29)

totensor = ToTensor()

# Let's build the model

# In[6]:

from rissl.models.e2_wide_resnet import e2wrn28_10R, e2wrn28_7R
from rissl.models.resnet import ResNet, BasicBlock


if args.model_type == "e2_resnet_incorrect":
    model = e2_resnet18(N=args.N, F=1.0, sigma=0.45, restrict=-1, flip=False, fixparams=False, num_classes=10, num_input_channels=3, conv2triv=False).to(device)
elif args.model_type == "e2_wideresnet_correct":
    model = e2wrn28_7R(N=args.N, F=1.0, sigma=0.45, r=-1, fixparams=False, num_classes=10, num_channels=1).to(device)
else:
    assert False

print("Model:")
print(model)

# The model is now randomly initialized. 
# Therefore, we do not expect it to produce the right class probabilities.
# 
# However, the model should still produce the same output for rotated versions of the same image.
# This is true for rotations by multiples of $\frac{\pi}{2}$, but is only approximate for rotations by $\frac{\pi}{4}$.
# 
# Let's test it on a random test image:
# we feed eight rotated versions of the first image in the test set and print the output logits of the model for each of them.


def test_model(model: torch.nn.Module, x: Image):
    # evaluate the `model` on 8 rotated versions of the input image `x`
    model.eval()

    x = pad(x)
    
    print()
    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(10)])
    print(header)
    with torch.no_grad():
        for r in range(8):
            x_transformed = totensor(x.rotate(r*45., Image.BILINEAR)).reshape(1, 1, 29, 29)
            x_transformed = x_transformed.to(device)

            y = model(x_transformed)
            y = y.to('cpu').numpy().squeeze()
            
            angle = r * 45
            print("{:5d} : {}".format(angle, y))
    print('##########################################################################################')
    print()


# build the test set
if args.dataset == "mnist":
    test_set = MnistRotDataset(mode='test')
else:
    raise NotImplementedError

# retrieve the first image from the test set
x, y = next(iter(test_set))

# evaluate the model
test_model(model, x)

# TODO finish up that part
if args.train:

    train_transform = Compose([
        pad,
        RandomRotation(180, resample=Image.BILINEAR, expand=False),
        totensor,
    ])

    mnist_train = MnistRotDataset(mode='train', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64)


    test_transform = Compose([
        pad,
        totensor,
    ])
    mnist_test = MnistRotDataset(mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)



    from tqdm import tqdm
    for epoch in tqdm(range(5)):
        model.train()
        for i, (x, t) in enumerate(train_loader):

            optimizer.zero_grad()

            x = x.to(device)
            t = t.to(device)

            y = model(x)

            loss = loss_function(y, t)

            loss.backward()

            optimizer.step()

        if epoch % 10 == 0:
            total = 0
            correct = 0
            with torch.no_grad():
                model.eval()
                for i, (x, t) in enumerate(test_loader):

                    x = x.to(device)
                    t = t.to(device)

                    y = model(x)

                    _, prediction = torch.max(y.data, 1)
                    total += t.shape[0]
                    correct += (prediction == t).sum().item()
            print(f"epoch {epoch} | test accuracy: {correct/total*100.}")


    # retrieve the first image from the test set
    x, y = next(iter(test_set))


    # evaluate the model
    test_model(model, x)

