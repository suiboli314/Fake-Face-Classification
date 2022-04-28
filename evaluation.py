import click
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from models import load_model

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


@click.command()
@click.option('--save_dir', 'save_dir',                 help='direcory to save the result', type=str, default="./save/", required=True)
@click.option('--model_id', 'model_id',                 help='id of saved model', type=str, default="newModel000", required=True)
@click.option('--dataset_dir', 'dataset_dir',           help='dataset directory', type=str, default="./Fake-Face-Classification/fake_real-faces/", required=True)
@click.option('--model_name', 'model_name',             help='model type, e.g. regnet', type=str, default="regnet", required=True)
@click.option('--img_size', 'img_size',                 help='size of image in dataset', type=int, default=199, required=True)
@click.option('--batch_size', 'batch_size',             help='size of each batch', type=int, default=2, required=True)
def test(save_dir, model_id, dataset_dir, model_name, img_size, batch_size):
    
    # load model
    model = load_model(model_name)
    model.load_state_dict(torch.load(join(save_dir,'best_{}.pth'.format(model_id))))
    model = model.to(device)

    # load test dataset
    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    test_data = datasets.ImageFolder(
        join(dataset_dir, '/test'), transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # criterion
    criterion = nn.CrossEntropyLoss()


    # test
    model.eval()
    loss_ = 0
    valid_acc = 0
    num_image = 0
    pred = []
    true = []
    pred_wrong = []
    true_wrong = []
    image = []
    for x, y_true in test_loader:
        X = x.to(device)
        Y = y_true.to(device)
        logit = model(X)
        loss = criterion(logit.squeeze(1), Y.long())
        loss_ += loss.item() * x.size(0)
        Max, num = torch.max(logit, 1)
        valid_acc += torch.sum(num == Y)
        num_image += x.size(0)

    preds = num.numpy()
    target = Y.numpy()
    preds = np.reshape(preds, (len(preds), 1))
    target = np.reshape(target, (len(preds), 1))
    for i in range(len(preds)):
        pred.append(preds[i])
        true.append(target[i])
        if (preds[i] != target[i]):
            pred_wrong.append(preds[i])
            true_wrong.append(target[i])
            image.append(x[i])

    total_loss_valid = loss_ / num_image
    total_acc_valid = valid_acc / num_image
    print(f"Accuracy: {total_acc_valid.item()}")
    print(f"Loss: {total_loss_valid}")
    return true, pred, image, true_wrong, pred_wrong, total_loss_valid, total_acc_valid.item(), model


def performance_matrix(true, pred, model):
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, average='macro')
    print('Confusion Matrix:\n', metrics.confusion_matrix(true, pred))
    print('Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}'.format(precision * 100, recall * 100, accuracy * 100,
                                                                         f1_score * 100))
    #metrics.plot_confusion_matrix(model, true, pred)

if __name__ == '__main__':
    true, pred, image, true_wrong, pred_wrong, total_loss_valid, total_acc_valid, model = test()
    performance_matrix(true, pred, model)