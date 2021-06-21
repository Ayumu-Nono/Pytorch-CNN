import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

from block import Block
from model import ResNet50, GlobalAvgPool2d


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    データの読み込み
    '''
    root = os.path.join(os.path.dirname(__file__),
                        '..', 'data', 'fashion_mnist')
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = \
        torchvision.datasets.FashionMNIST(root=root,
                                          download=True,
                                          train=True,
                                          transform=transform)
    mnist_test = \
        torchvision.datasets.FashionMNIST(root=root,
                                          download=True,
                                          train=False,
                                          transform=transform)

    train_dataloader = DataLoader(mnist_train,
                                  batch_size=100,
                                  shuffle=True)
    test_dataloader = DataLoader(mnist_test,
                                 batch_size=100,
                                 shuffle=False)

    '''
    モデルの構築
    '''
    model = ResNet50(10).to(device)

    '''
    モデルの学習・評価
    '''
    def compute_loss(label, pred):
        return criterion(pred, label)

    def train_step(x, t):
        model.train()
        preds = model(x)
        loss = compute_loss(t, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, preds

    def test_step(x, t):
        model.eval()
        preds = model(x)
        loss = compute_loss(t, preds)

        return loss, preds

    criterion = nn.NLLLoss()
    optimizer = optimizers.Adam(model.parameters(), weight_decay=0.01)
    epochs = 5

    for epoch in range(epochs):
        train_loss = 0.
        test_loss = 0.
        test_acc = 0.

        for (x, t) in train_dataloader:
            x, t = x.to(device), t.to(device)
            loss, _ = train_step(x, t)
            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        for (x, t) in test_dataloader:
            x, t = x.to(device), t.to(device)
            loss, preds = test_step(x, t)
            test_loss += loss.item()
            test_acc += \
                accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print('Epoch: {}, Valid Cost: {:.3f}, Valid Acc: {:.3f}'.format(
            epoch+1,
            test_loss,
            test_acc
        ))