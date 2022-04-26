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
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import time

from datasets import get_transforms, get_dataset
import utils
from models import get_model

def train(local_rank, args):
    if local_rank == 0:
        # set up logging to file
        utils.setup_logging(args.log_dir)

        # set up tensorboard
        tb_path = os.path.join(args.log_dir, "tb_logs")
        os.makedirs(tb_path)
        writer = SummaryWriter(log_dir=tb_path)

        # create directory for model checkpoints
        checkpoint_path = os.path.join(args.log_dir, "checkpoints")
        os.makedirs(checkpoint_path)
        best_model_path = os.path.join(checkpoint_path, "best_model.pt")
        final_model_path = os.path.join(checkpoint_path, "final_model.pt")

    print('local rank:', local_rank)
    # rank calculation for each process per gpu so that they can be identified uniquely.
    global_rank = int(os.environ.get("SLURM_NODEID")) * args.ngpus_per_node + local_rank
    print('global rank:', global_rank)

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=global_rank
    )

    # fix seed
    utils.fix_seed(args.seed)

    train_transform, test_transform = get_transforms(args)

    if local_rank == 0:
        args.transform = utils.add_transform_to_args(train_transform, test_transform)

        utils.setup_mlflow(args)

        # save args to json
        args_path = os.path.join(args.log_dir, "args.json")
        with open(args_path, 'w') as file:
            json.dump(vars(args), file, indent=4)

    train_dataset, test_dataset, num_classes = get_dataset(train_transform, test_transform, args)

    if local_rank == 0:
        logging.info(f"Train size {len(train_dataset)}, test size {len(test_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=global_rank)
    # remove `shuffle` argument when using `sampler`
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 4, shuffle=False, num_workers=args.num_workers)

    device = torch.device(local_rank)
    logging.info(f"Device: {device}")
    mlflow.log_param("device", device)

    # select a model with randomly initialized weights, default is resnet18 so that we can train it quickly
    model = get_model(args.model_type, num_classes, args).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.max_lr, momentum=args.momentum)

    if args.lr_scheduler_type == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=args.max_lr,
                                                        div_factor=args.max_lr / args.start_lr,
                                                        # div_factor = max_lr / start_lr
                                                        final_div_factor=args.start_lr / args.end_lr,
                                                        # final_div_factor = start_lr / end_lr
                                                        pct_start=args.lr_pct_start,
                                                        steps_per_epoch=len(train_loader),
                                                        epochs=args.num_epochs)
    elif args.lr_scheduler_type == "StepLR":
        # step_size and gamma from Worrall and Welling, 2019
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    else:
        raise ValueError("Incorrect type of learning rate scheduler.")

    train_losses = []
    train_accs = []

    if not args.no_validation:
        best_test_loss = np.inf
        best_test_acc = 0.0
        best_test_epoch = 0
        test_losses = []
        test_accs = []

    for epoch_idx in range(args.num_epochs):  # loop over the dataset multiple times
        if local_rank == 0:
            logging.info(f"Epoch {epoch_idx}")

        model.train()

        train_epoch_loss = 0.0
        num_correct = 0.0
        actual_train_size = 0

        if args.profile and local_rank == 0:
            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=10, warmup=10, active=50, repeat=4),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_path),
                record_shapes=True,
                with_stack=True)
            prof.start()

        start_load = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch_load_ms = (time.time() - start_load) * 1000.0
            if local_rank == 0:
                writer.add_scalar("Speed/batch_load_ms", batch_load_ms, batch_idx)
            inputs, labels = batch
            batch_size = inputs.shape[0]

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)

            batch_loss.backward()
            optimizer.step()

            global_step = epoch_idx * len(train_loader) + batch_idx
            if local_rank == 0:
                writer.add_scalar("learning_rate/lr_per_batch", optimizer.param_groups[0]["lr"], global_step)

            if args.lr_scheduler_type == "OneCycleLR":
                scheduler.step()

            train_epoch_loss += batch_size * batch_loss.item()
            actual_train_size += batch_size
            preds = outputs.argmax(dim=1)
            num_correct += (preds == labels).sum().item()

            start_load = time.time()

            if args.profile and local_rank == 0:
                prof.step()

        if args.profile and local_rank == 0:
            prof.stop()

        if args.lr_scheduler_type == "StepLR":
            scheduler.step()

        # based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#test-the-network-on-the-test-data
        # explicitly calculating actual train size instead of using len(train_dataset) is more robust
        # because if we drop the last batch, those two are not equal
        train_epoch_loss = train_epoch_loss / actual_train_size
        train_epoch_acc = 100 * num_correct / actual_train_size

        if local_rank == 0:
            writer.add_scalar("train/epoch_loss", train_epoch_loss, epoch_idx)
            writer.add_scalar("train/epoch_acc", train_epoch_acc, epoch_idx)
            writer.add_scalar("learning_rate/lr_per_epoch", optimizer.param_groups[0]["lr"], epoch_idx)

        if epoch_idx % args.save_model_every_n_epochs == 0 and local_rank == 0:
            # save model every n epochs
            epoch_model_path = os.path.join(checkpoint_path, f"model_epoch_{epoch_idx}.pt")
            logging.info(f"Saving the model at epoch {epoch_idx} to {epoch_model_path} ")
            torch.save(model.state_dict(), epoch_model_path)

        train_losses.append(train_epoch_loss)
        train_accs.append(train_epoch_acc)

        # evaluate the model on validation set
        if not args.no_validation and local_rank == 0:
            logging.info(f"Starting validation")
            test_epoch_loss = 0.0
            test_num_correct = 0.0
            actual_test_size = 0.0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for batch in test_loader:
                    model.eval()

                    inputs, labels = batch
                    batch_size = inputs.shape[0]

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # calculate outputs by running images through the network
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels)

                    test_epoch_loss += batch_size * batch_loss.item()
                    actual_test_size += batch_size
                    preds = outputs.argmax(dim=1)
                    test_num_correct += (preds == labels).sum().item()

            test_epoch_loss = test_epoch_loss / actual_test_size
            test_epoch_acc = 100 * test_num_correct / actual_test_size

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
                logging.info(
                    f"Early stopping at epoch {epoch_idx} - test loss haven't improved for {args.patience} epochs")
                break

            logging.info("Finished validation")

    if not args.no_validation and local_rank == 0:
        mlflow.log_metric("best_test_loss", best_test_loss)
        mlflow.log_metric("best_test_acc", best_test_acc)
        mlflow.log_metric("best_test_epoch", best_test_epoch)

        mlflow.log_metric("final_test_loss", test_epoch_loss)
        mlflow.log_metric("final_test_acc", test_epoch_acc)

    if local_rank == 0:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = utils.parse_args(parser)

    # Total number of gpus availabe to us.
    args.world_size = args.ngpus_per_node * args.num_nodes
    # add the ip address to the environment variable so it can be easily avialbale
    os.environ['MASTER_ADDR'] = args.ip_address
    logging.debug(f"Master IP address is {args.ip_address}")
    logging.info(f"Number of devices {torch.cuda.device_count()}")
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = str(args.world_size)

    logging.info(f"Start training")
    # nprocs: number of process which is equal to args.ngpu here
    mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))

    logging.info('Finished training')