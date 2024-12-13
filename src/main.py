import torch
import torch.nn as nn
import torchvision

from dataloader import get_dataloader, load_dataset
from models.lstm import LSTM
from models.rnn import RNN
from trainer import Trainer


def create_dataloader(folder_path_train, folder_path_test, batch_size=32):
    train_dataset, val_dataset, test_dataset = load_dataset(
        folder_path_train, folder_path_test
    )
    train_dataloader = get_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def get_cnn_pretrained(model: str = "resnet50" or "resnet101" or "resnet152"):
    if model == "resnet50":
        CNN_pretrained = torchvision.models.resnet50(pretrained=True)
    elif model == "resnet101":
        CNN_pretrained = torchvision.models.resnet101(pretrained=True)
    else:
        CNN_pretrained = torchvision.models.resnet152(pretrained=True)
    CNN_pretrained = torch.nn.Sequential(*list(CNN_pretrained.children())[:8])
    return CNN_pretrained


def get_trainer(model, train_dataloader, val_dataloader):
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 100

    return Trainer(
        train_dataloader, val_dataloader, model, optimizer, criterion, n_epochs
    )


def main():
    folder_path_train = "../video_data/train/"
    folder_path_test = "../video_data/test/"

    HIDDEN_SIZE = 512
    NUM_CLASSES = 5
    BATCH_SIZE = 32
    train_dataloader, val_dataloader, test_dataloader = create_dataloader(
        folder_path_train, folder_path_test, BATCH_SIZE
    )
    print("get data done")

    CNNPretrained = get_cnn_pretrained("resnet101")
    print("get model done")
    model = RNN(CNNPretrained, HIDDEN_SIZE, NUM_CLASSES, BATCH_SIZE)
    trainer = get_trainer(model, train_dataloader, val_dataloader)
    print("get trainer done")
    path_to_save = "model.pth"
    trainer.train(path_to_save)


if __name__ == "__main__":
    main()
