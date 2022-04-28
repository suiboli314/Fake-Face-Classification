import time
import click
import torch
from torch import nn
from adabelief_pytorch import AdaBelief

from models import load_model
from datasets.dataset_torch import Dataset
from utils.hp import load_hps
from utils.plotting import plot
from utils.callback import CSV_log, Model_checkpoint

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train(ds_train, model, criterion, optimizer, device):
    model.train()
    loss_ = 0
    train_acc = 0
    num_image = 0
    for x, y_true in ds_train:
        optimizer.zero_grad()
        X = x.to(device)
        Y = y_true.to(device)
        logit = model(X)
        loss = criterion(logit.squeeze(1), Y.long())
        loss_ += loss.item() * x.size(0)
        Max, num = torch.max(logit, 1)
        train_acc += torch.sum(num == Y)
        num_image += x.size(0)
        loss.backward()
        optimizer.step()
    total_loss_train = loss_ / num_image
    total_acc_train = train_acc / num_image

    return model, total_loss_train, total_acc_train.item()


def valid(ds_valid, model, criterion, device):
    model.eval()
    loss_ = 0
    valid_acc = 0
    num_image = 0
    for x, y_true in ds_valid:
        X = x.to(device)
        Y = y_true.to(device)
        logit = model(X)
        loss = criterion(logit.squeeze(1), Y.long())
        loss_ += loss.item() * x.size(0)
        Max, num = torch.max(logit, 1)
        valid_acc += torch.sum(num == Y)
        num_image += x.size(0)
    total_loss_valid = loss_ / num_image
    total_acc_valid = valid_acc / num_image
    return model, total_loss_valid, total_acc_valid.item()


def training(model, ds_train, ds_valid, criterion, optimizer, scheduler, device, epochs, save_dir="./save/", model_id="newModel000"):
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    history = {}

    name_model = 'best_{}.pth'.format(model_id)
    name_csv = 'log_{}.csv'.format(model_id)

    for epoch in range(epochs):
        since = time.time()
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)
        # training
        model, total_loss_train, total_acc_train = train(
            ds_train, model, criterion, optimizer, device)
        train_losses.append(total_loss_train)
        train_accs.append(total_acc_train)
        # validation
        with torch.no_grad():
            model, total_loss_valid, total_acc_valid = valid(
                ds_valid, model, criterion, device)
            valid_losses.append(total_loss_valid)
            valid_accs.append(total_acc_valid)

        scheduler.step(total_loss_valid)
        metrics = {'train_loss': train_losses, 'train_acc': train_accs, 'val_loss': valid_losses,
                   'val_acc': valid_accs}

        # save checkpoint
        Model_checkpoint(path=save_dir, metrics=metrics, model=model,
                         monitor='val_acc', verbose=True,
                         file_name=name_model)
        # printout
        history = {'epoch': epoch, 'accuracy': train_accs, 'loss': train_losses, 'val_accuracy': valid_accs,
                   'val_loss': valid_losses, 'LR': optimizer.param_groups[0]['lr']}
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Epoch:", epoch + 1, "- Train Loss:", total_loss_train, "- Train Accuracy:", total_acc_train,
              "- Validation Loss:", total_loss_valid, "- Validation Accuracy:", total_acc_valid)
        # save log
        CSV_log(path=save_dir, score=history, filename=name_csv)

    return model, history


@click.command()
@click.option('--save_dir', 'save_dir',                 help='direcory to save the result', type=str, default="./save/", required=True)
@click.option('--model_id', 'model_id',                 help='id of saved model', type=str, default="newModel000", required=True)
@click.option('--dataset_dir', 'dataset_dir',           help='dataset directory', type=str, default="./Fake-Face-Classification/fake_real-faces/", required=True)
@click.option('--model_name', 'model_name',             help='model type, e.g. regnet', type=str, default="regnet", required=True)
@click.option('--n_epochs', 'n_epochs',                 help='number of epoch', type=int, default=50, required=True)
@click.option('--batch_size', 'batch_size',             help='size of each batch', type=int, default=2, required=True)
@click.option('--lr', 'learning_rate',                  help='learning rate', type=float, default=0.001, required=True)
@click.option('--lr_factor', 'lr_reducer_factor',       help='factor for learning rate reducer', type=float, default=0.2, required=True)
@click.option('--lr_patience', 'lr_reducer_patience',   help='patience of learning rate reducer', type=int, default=8, required=True)
@click.option('--img_size', 'img_size',                 help='size of image in dataset', type=int, default=299, required=True)
# @click.option('--framework', 'framework', help='machine learning framework', required=True)
def train_torch(dataset_dir: str,
                model_id: str,
                save_dir: str,
                model_name: str,
                n_epochs: int,
                batch_size: int,
                learning_rate: float,
                lr_reducer_factor: float,
                lr_reducer_patience: int,
                img_size: int):

    hps = load_hps(dataset_dir=dataset_dir,
                   model_name=model_name,
                   n_epochs=n_epochs,
                   batch_size=batch_size,
                   learning_rate=learning_rate,
                   lr_reducer_factor=lr_reducer_factor,
                   lr_reducer_patience=lr_reducer_patience,
                   img_size=img_size)

    model = load_model(model_name=hps['model_name'])

    # if hps['framework'] == 'pytorch':
    train_loader, val_loader, test_loader = Dataset.pytorch_preprocess(dataset_dir=hps['dataset_dir'],
                                                                       img_size=hps['img_size'],
                                                                       batch_size=hps['batch_size'],
                                                                       split_size=0.3, augment=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdaBelief(model.parameters(), lr=hps['learning_rate'])
    reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   factor=hps['lr_reducer_factor'],
                                                                   patience=hps['lr_reducer_patience'],
                                                                   verbose=True)
    model, history = training(model, train_loader, val_loader, criterion, optimizer,
                              reduce_on_plateau, device, hps['n_epochs'], save_dir=save_dir, model_id=model_id)
    plot(history)


if __name__ == '__main__':
    train_torch()
