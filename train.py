import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
from torchvision import transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import mlflow
import json

from datasets.breakhis_fold_dataset import BreakhisFoldDataset
from datasets.breakhis_dataset import BreakhisDataset
from utils import setup_logging, parse_args, fix_seed
from models import models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    job_id = args.log_dir.split("/")[-1]
    mlflow.set_tracking_uri(f"file:///{args.mlflow_dir}")
    mlflow.set_experiment(args.exp_name)
    mlflow.start_run(run_name=job_id)
    mlflow.log_params(vars(args))
    setup_logging(args.log_dir)

    # save args to json
    args_path = os.path.join(args.log_dir, "args.json")
    with open(args_path, 'w') as file:
        json.dump(vars(args), file, indent=4)

    # fix seed
    fix_seed(args.seed)

    # same transforms as for supervised equivariant networks
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], # statistics from ImageNet
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
    model = models[args.model_type](**model_args).to(device)

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

    checkpoint_path = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(checkpoint_path)
    best_model_path = os.path.join(checkpoint_path, "best_model.pt")
    final_model_path = os.path.join(checkpoint_path, "final_model.pt")

    train_losses = []
    train_accs = []

    if not args.no_validation:
        best_test_loss = 1000.0
        best_test_acc = 0.0
        best_test_epoch = 0
        test_losses = []
        test_accs = []

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
            outputs = model(inputs).squeeze() # use squeeze to make the shape compatible with the loss
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

        train_losses.append(train_epoch_loss)
        train_accs.append(train_epoch_acc)

        # evaluate the model on validation set
        if not args.no_validation:
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

            test_losses.append(test_epoch_loss)
            test_accs.append(test_epoch_acc)

            logging.info(f"Train loss: \t{train_epoch_loss}, train acc: \t{train_epoch_acc}")
            logging.info(f"Test loss: \t{test_epoch_loss}, test acc: \t{test_epoch_acc}")

            # set new best loss, acc and epoch and save model
            if test_epoch_acc > best_test_acc:
                best_test_loss = test_epoch_loss
                best_test_acc = test_epoch_acc
                best_test_epoch = epoch_idx

                # save parameters of the best model for inference
                logging.info(f"Saving the best model to {best_model_path}")
                torch.save(model.state_dict(), best_model_path)

            # if using early stopping, end when loss didn't improve for n epochs
            elif args.early_stopping and epoch_idx - best_test_epoch >= args.patience:
                logging.info(f"Early stopping at epoch {epoch_idx} - test loss haven't improved for {args.patience} epochs")
                break

    if not args.no_validation:
        mlflow.log_metric("best_test_loss", best_test_loss)
        mlflow.log_metric("best_test_acc", best_test_acc)
        mlflow.log_metric("best_test_epoch", best_test_epoch)

        mlflow.log_metric("final_test_loss", test_epoch_loss)
        mlflow.log_metric("final_test_acc", test_epoch_acc)

    # save metrics summary
    metrics_summary_path = os.path.join(args.log_dir, "metrics_summary.json")
    metrics_summmary = {"best_test_loss": best_test_loss,
                        "best_test_acc": best_test_acc,
                        "best_test_epoch": best_test_epoch,
                        "final_test_loss": test_epoch_loss,
                        "final_test_acc": test_epoch_acc
                        }

    with open(metrics_summary_path, 'w') as file:
        json.dump(metrics_summmary, file, indent=4)

    # save metrics over epochs to json
    metrics_path = os.path.join(args.log_dir, "metrics_over_epochs.json")
    metrics = {"train": {"losses": train_losses,
                         "accs": train_accs},
               "test": {"losses": test_losses,
                        "accs": test_accs}
               }

    with open(metrics_path, 'w') as file:
        json.dump(metrics, file, indent=4)

    # save parameters of the model after the last epoch for inference
    logging.info(f"Saving the final model to {final_model_path}")
    torch.save(model.state_dict(), final_model_path)

    writer.close()
    logging.info('Finished training')