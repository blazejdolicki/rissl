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
import mlflow
import random

from datasets.breakhis_fold_dataset import BreakhisFoldDataset
from datasets.breakhis_dataset import BreakhisDataset
from utils import setup_logging, parse_args, fix_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    job_id = args.log_dir.split("/")[-1]
    mlflow.set_tracking_uri(f"file:///{args.mlflow_dir}")
    mlflow.set_experiment(args.exp_name)
    mlflow.start_run(run_name=job_id)
    mlflow.log_params(vars(args))
    setup_logging(args.log_dir)

    # fix seed
    fix_seed(args.seed)

    # binary classification
    archs = {"resnet18": models.resnet18,
             "resnet34": models.resnet34,
             "resnet50": models.resnet50,
             "resnet101": models.resnet101,
             "resnet152": models.resnet152,
             "resnext50_32x4d": models.resnext50_32x4d,
             "resnext101_32x8d": models.resnext101_32x8d,
             "wide_resnet50_2": models.wide_resnet50_2,
             "wide_resnet101_2": models.wide_resnet101_2,
             }

    assert args.model_type in archs, \
        f"Model {args.model_type} is not supported. Choose one of the following models: {list(archs.keys())}"

    # same transforms as for supervised equivariant networks
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]),
                                    ])

    if args.dataset == "breakhis_fold":
        train_dataset = BreakhisFoldDataset(args.data_dir, "train", args.fold, args.train_mag, transform)
        test_dataset = BreakhisFoldDataset(args.data_dir, "test", args.fold, args.test_mag, transform)
    elif args.dataset == "breakhis":
        train_dataset = BreakhisDataset(args.data_dir, "train", transform)
        test_dataset = BreakhisDataset(args.data_dir, "test", transform)

    logging.info(f"Train size {len(train_dataset)}, test size {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # select a model with randomly initialized weights, default is resnet18 so that we can train it quickly
    model_args = {"num_classes": 1} # BCE loss
    model = archs[args.model_type](**model_args).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.max_lr, momentum=args.momentum)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=args.max_lr,
                                                    div_factor=args.max_lr/args.start_lr,  # div_factor = max_lr / start_lr
                                                    final_div_factor=args.start_lr/args.end_lr, # final_div_factor = start_lr / end_lr
                                                    pct_start=args.lr_pct_start,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=args.num_epochs)

    tb_path = os.path.join(args.log_dir, "tb_logs")
    os.makedirs(tb_path)
    writer = SummaryWriter(log_dir=tb_path)

    if args.early_stopping:
        best_test_loss = 1000.0
        best_test_acc = 0.0
        best_test_epoch = 0

    for epoch_idx in tqdm(range(args.num_epochs)):  # loop over the dataset multiple times
        logging.info(f"Epoch {epoch_idx}")

        model.train()

        train_epoch_loss = 0.0
        correct = 0.0
        total = 0
        for batch_idx, batch in enumerate(train_loader):
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

            global_step = epoch_idx * len(train_loader) + batch_idx
            writer.add_scalar("learning_rate/lr_per_batch", optimizer.param_groups[0]["lr"], global_step)
            scheduler.step()

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
        writer.add_scalar("learning_rate/lr_per_epoch", optimizer.param_groups[0]["lr"], epoch_idx)

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

        # set new best loss, acc and epoch
        if test_epoch_acc > best_test_acc:
            best_test_loss = test_epoch_loss
            best_test_acc = test_epoch_acc
            best_test_epoch = epoch_idx
        # if using early stopping, end when loss didn't improve for n epochs
        elif args.early_stopping and epoch_idx - best_test_epoch >= args.patience:
            logging.info(f"Early stopping at epoch {epoch_idx} - test loss haven't improved for {args.patience} epochs")
            break

    mlflow.log_metric("best_test_loss", best_test_loss)
    mlflow.log_metric("best_test_acc", best_test_acc)
    mlflow.log_metric("best_test_epoch", best_test_epoch)


    writer.close()
    logging.info('Finished training')