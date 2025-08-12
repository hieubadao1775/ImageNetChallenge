import os.path
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from model import ResNet50
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, ColorJitter, Normalize
from tqdm import tqdm
from dataset import Animal
from argparse import ArgumentParser
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="binary")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default=r"../data/animals")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--checkpoint", type=str, default="checkpoint")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--logdir", type=str, default="tensorboard")
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    return parser.parse_args()


def initialize_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_transform = Compose([
        # ColorJitter(),
        ToTensor(),
        Resize((224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    val_transform = Compose([
        ToTensor(),
        Resize((224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    train_data = Animal(root=os.path.join(args.root, "train"), transform=train_transform)
    val_data = Animal(root=os.path.join(args.root, "val"), transform=val_transform)

    train_params = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": True,
        "drop_last": True
    }
    val_params = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": False,
        "drop_last": False
    }
    train_dataloader = DataLoader(train_data, **train_params)
    val_dataloader = DataLoader(val_data, **val_params)

    model = ResNet50(len(train_data.categories)).to(device)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if not os.path.isdir(args.checkpoint):
        os.mkdir(args.checkpoint)
    if args.resume:
        checkpoint = torch.load(os.path.join(args.checkpoint, "last.pt"))
        model = model.load_state_dict(checkpoint["model"])
        optimizer = optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]
    else:
        start_epoch = 0
        best_accuracy = 0
        model.apply(initialize_weight)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    writer = SummaryWriter(args.logdir)
    iter_nums = len(train_dataloader)

    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        train_progressbar = tqdm(train_dataloader, colour="green")
        for i, (images, labels) in enumerate(train_progressbar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_progressbar.set_description(f"Epoch {epoch + 1}/{args.epochs}. Iter {i}/{iter_nums}. Loss {loss:.6f}")
            writer.add_scalar("Train/Loss", loss, epoch * iter_nums + i)

        # Val
        model.eval()
        val_progressbar = tqdm(val_dataloader, colour="green")
        list_labels = []
        list_predictions = []
        for images, labels in val_progressbar:
            images = images.to(device)

            with torch.no_grad():
                predictions= model(images)
                predictions = torch.argmax(predictions, dim=1).cpu().detach().numpy()
            list_predictions.extend(predictions)
            list_labels.extend(labels.numpy())
        accuracy = accuracy_score(list_labels, list_predictions)
        print("Accuracy:", accuracy)

        writer.add_scalar("Accuracy", accuracy, epoch)
        plot_confusion_matrix(writer, confusion_matrix(list_labels, list_predictions), class_names=val_data.categories,
                              epoch=epoch)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_accuracy": accuracy if accuracy > best_accuracy else best_accuracy
        }
        torch.save(checkpoint, f"{args.checkpoint}/last.pt")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_accuracy": accuracy if accuracy > best_accuracy else best_accuracy
            }
            torch.save(checkpoint, f"{args.checkpoint}/best.pt")

        # scheduler.step(accuracy)
        scheduler.step()


if __name__ == '__main__':
    args = get_args()
    train(args)