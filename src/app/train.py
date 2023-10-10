#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import argparse
from pathlib import Path

from model import *
import dataset
import evalutation

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a simple CNN on CIFAR-10",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / "data"
parser.add_argument("--data-dir", default=default_dataset_dir, type=Path)
parser.add_argument("--log-dir", default=Path("tensorboard_logs"), type=Path)
parser.add_argument("--fig-dir", default=Path("figs"), type=Path)
parser.add_argument(
    "--learning-rate", default=0.00005, type=float, help="Learning rate"
)
parser.add_argument("--l1-alpha", default=0.0001, type=float, help="L1 alpha")
parser.add_argument("--net", default="shallow", type=str)
parser.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=200,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=1,
    type=int,
    help="Number of worker processes used to load data.",
)


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    train_dataset = dataset.GTZAN("data/train.pkl")
    test_dataset = dataset.GTZAN("data/val.pkl")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    # select model from command-line arguments
    if args.net == "shallow":
        model = ShallowNet(class_count=10)
    elif args.net == "deep":
        model = DeepNet(class_count=10)
    else:
        model = BBNN(class_count=10)

    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    log_dir = get_summary_writer_log_dir(args)  # directory to store tensorboard logs
    fig_dir = get_summary_writer_fig_dir(args)  # directory to store confusion matrices
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(str(log_dir), flush_secs=5)

    start_epoch = 0
    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        summary_writer,
        DEVICE,
        fig_dir,
    )
    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
        start_epoch=start_epoch,
        l1_alpha=args.l1_alpha,
    )

    summary_writer.close()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
        fig_dir: Path,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.fig_dir = fig_dir
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0,
        l1_alpha=0.0001,
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for filenames, batch, labels, samples in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                logits = self.model(batch)
                weights = torch.cat(
                    [
                        p.view(-1)
                        for n, p in self.model.named_parameters()
                        if ".weight" in n
                    ]
                )
                l1_penalty = torch.sum(
                    torch.abs(weights)
                )  # generate l1 penalty based on absolute weight values
                loss = self.criterion(logits, labels) + (l1_alpha * l1_penalty)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)

            if ((epoch + 1) % val_frequency) == 0:
                save_confusion = ((epoch + 1) % 100) == 0
                self.validate(save_confusion=save_confusion)
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
            f"epoch: [{epoch}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"batch loss: {loss:.5f}, "
            f"batch accuracy: {accuracy * 100:2.2f}, "
            f"data load time: "
            f"{data_load_time:.5f}, "
            f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars("accuracy", {"train": accuracy}, self.step)
        self.summary_writer.add_scalars(
            "loss", {"train": float(loss.item())}, self.step
        )
        self.summary_writer.add_scalar("time/data", data_load_time, self.step)
        self.summary_writer.add_scalar("time/data", step_time, self.step)

    def save_confusion_matrix(self, results):
        classes = (
            "blues",
            "classical",
            "country",
            "disco",
            "hiphop",
            "jazz",
            "metal",
            "pop",
            "reggae",
            "rock",
        )
        c_matrix = confusion_matrix(results["labels"], results["preds"])
        df_c_matrix = pd.DataFrame(
            c_matrix / np.sum(c_matrix) * 10, index=[i for i in classes]
        )
        plt.figure(figsize=(12, 7))
        sn.heatmap(
            df_c_matrix,
            annot=True,
            xticklabels=classes,
            yticklabels=classes,
            vmin=0,
            vmax=1,
            square=True,
        )
        plt.savefig(f"{self.fig_dir}_{self.step}.png")

    def validate(self, save_confusion=False):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for filenames, batch, labels, samples in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        # raw_accuracy = evalutation.evaluate(preds, "data/val.pkl", self.device)
        average_loss = total_loss / len(self.val_loader)

        if save_confusion:
            self.save_confusion_matrix(results)

        self.summary_writer.add_scalars("accuracy", {"test": accuracy}, self.step)
        self.summary_writer.add_scalars("loss", {"test": average_loss}, self.step)
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


#
def dir_prefix(args):
    """Generate the prefix to which run data such as
        tensorboard logs and cunfusion matrices are stored

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory name for a given set of hyperparameters. Does not include run number.
    """
    return (
        f"CNN_bn_"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        f"net={args.net}_"
        f"l1_alpha={args.l1_alpha}_"
        f"run_"
    )


def get_checkpoint_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = dir_prefix(args)
    i = 0
    while i < 1000:
        tb_log_dir = args.checkpoint_path / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = dir_prefix(args)
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


def get_summary_writer_fig_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been figged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of fig_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_fig_dir_prefix = dir_prefix(args)
    i = 0
    while i < 1000:
        tb_fig_dir = args.fig_dir / (tb_fig_dir_prefix + str(i))
        if not tb_fig_dir.exists():
            return str(tb_fig_dir)
        i += 1
    return str(tb_fig_dir)


if __name__ == "__main__":
    main(parser.parse_args())
