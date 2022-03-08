import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import logging
from torchvision import transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

from breakhis_dataset import BreakhisDataset
from utils import setup_logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/bdolicki/thesis/ssl-histo/data/breakhis',
                        help='Directory of the data')
    parser.add_argument('--fold', type=int, default=1, choices=[1, 2, 3, 4, 5], help='Fold used for training and testing')
    parser.add_argument('--train_mag', type=str, default=40, choices=["40", "100", "200", "400"],
                        help='Magnitude of training images')
    parser.add_argument('--test_mag', type=str, default=40, choices=["40", "100", "200", "400"],
                        help='Magnitude of testing images')
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers used to load the data")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--model_type', type=str, default='resnet18', help='Model architecture')
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum of stochastic gradient descent")
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of training epochs')

    parser.add_argument('--seed', type=int, default=7, help='Random seed')
    parser.add_argument('--output_dir', type=str, default="logs", help='Output directory for logging')
    args = parser.parse_args()

    # binary classification
    archs = {"resnet18": models.resnet18,
             "resnet34": models.resnet34,
             "resnet50": models.resnet50,
             "resnet101": models.resnet101,
             "resnet152": models.resnet152,
             "resnext50_32x4d": models.resnext50_32x4d,
             "resnext101_32x8d": models.resnext101_32x8d,
             "wide_resnet50_2": models.wide_resnet50_2,
             "wide_resnet101_2": models.wide_resnet101_2
             }

    assert args.model_type in archs, \
        f"Model {args.model_type} is not supported. Choose one of the following models: {list(archs.keys())}"

    setup_logging(args.output_dir)

    # fix seed
    logging.info(f"Random seed: {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # same transforms as for supervised equivariant networks
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]),
                                    ])

    train_dataset = BreakhisDataset(args.data_dir, "train", args.fold, args.train_mag, transform)
    test_dataset = BreakhisDataset(args.data_dir, "test", args.fold, args.test_mag, transform)

    logging.info(f"Train size {len(train_dataset)}, test size {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # select a model with randomly initialized weights, default is resnet18 so that we can train it quickly
    model_args = {"num_classes": 1} # BCE loss
    model = archs[args.model_type](**model_args).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_logs"))

    for epoch_idx in tqdm(range(args.num_epochs)):  # loop over the dataset multiple times
        logging.info(f"Epoch {epoch_idx}")

        model.train()

        train_epoch_loss = 0.0
        correct = 0.0
        total = 0
        for idx, batch in enumerate(train_loader):
            inputs, labels = batch
            batch_size = inputs.shape[0]

            inputs = inputs.to(device)
            labels = labels.to(device).float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs).squeeze()
            batch_loss = criterion(outputs, labels)

            batch_loss.backward()
            optimizer.step()
            # log metrics
            train_epoch_loss += batch_size * batch_loss.item()
            total += batch_size
            # Note: for label equal to 1, we want outputs of the sigmoid to be above 0.5
            # which is equivalent to outputs before the sigmoid that are above 0
            # and since we are using BCEWithLogits loss, the outputs are not passed through sigmoid
            correct += ((outputs > 0.0) == labels).sum().item()

        # based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#test-the-network-on-the-test-data
        # explicitly calculating total instead of using len(train_dataset) is more robust
        # because if we drop the last batch, those two are not equal
        train_epoch_loss = train_epoch_loss / total
        train_epoch_acc = 100 * correct / total

        writer.add_scalar("train/epoch_loss", train_epoch_loss, epoch_idx)
        writer.add_scalar("train/epoch_acc", train_epoch_acc, epoch_idx)

        test_epoch_loss = 0.0
        test_correct = 0.0
        test_total = 0.0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for batch in test_loader:
                model.eval()

                inputs, labels = batch
                batch_size = inputs.shape[0]

                inputs = inputs.to(device)
                labels = labels.to(device).float()

                # calculate outputs by running images through the network
                outputs = model(inputs).squeeze()
                batch_loss = criterion(outputs, labels)

                test_epoch_loss += batch_size * batch_loss.item()
                test_total += batch_size
                test_correct += ((outputs > 0.0) == labels).sum().item()

        test_epoch_loss = test_epoch_loss / test_total
        test_epoch_acc = 100 * test_correct / test_total

        writer.add_scalar("test/epoch_loss", test_epoch_loss, epoch_idx)
        writer.add_scalar("test/epoch_acc", test_epoch_acc, epoch_idx)
        logging.info(f"Train loss: \t{train_epoch_loss}, train acc: \t{train_epoch_acc}")
        logging.info(f"Test loss: \t{test_epoch_loss}, test acc: \t{test_epoch_acc}")

    writer.close()
    logging.info('Finished training')