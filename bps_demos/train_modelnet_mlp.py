import numpy as np
import os
import sys

# PyTorch dependencies
import torch as pt
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

# local dependencies
from bps import bps
from modelnet40 import load_modelnet40


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, 'data')


N_BPS_POINTS = 512
BPS_RADIUS = 1.7
DEVICE = sys.argv[1]  # 'cpu' or 'cuda'


class ShapeClassifierMLP(nn.Module):

    def __init__(self, n_features, n_classes, hsize1=512,  hsize2=512, dropout1=0.8, dropout2=0.6):
        super(ShapeClassifierMLP, self).__init__()

        self.bn0 = nn.BatchNorm1d(n_features)
        self.fc1 = nn.Linear(in_features=n_features, out_features=hsize1)
        self.bn1 = nn.BatchNorm1d(hsize1)
        self.do1 = nn.Dropout(dropout1)
        self.fc2 = nn.Linear(in_features=hsize1, out_features=hsize2)
        self.bn2 = nn.BatchNorm1d(hsize2)
        self.do2 = nn.Dropout(dropout2)
        self.fc3 = nn.Linear(in_features=hsize2, out_features=n_classes)

    def forward(self, x):
        x = self.bn0(x)
        x = self.do1(self.bn1(F.relu(self.fc1(x))))
        x = self.do2(self.bn2(F.relu(self.fc2(x))))
        x = self.fc3(x)

        return x


def fit(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader, epoch_id):
    model.eval()
    test_loss = 0
    n_test_samples = len(test_loader.dataset)
    n_correct = 0
    with pt.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            n_correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= n_test_samples
    test_acc = 100.0 * n_correct / n_test_samples
    print(
        "Epoch {} test loss: {:.4f}, test accuracy: {}/{} ({:.2f}%)".format(epoch_id, test_loss, n_correct, n_test_samples, test_acc))

    return test_loss, test_acc


def main():

    # load modelnet point clouds
    xtr, ytr, xte, yte = load_modelnet40(root_data_dir=DATA_PATH)

    # this will normalise your point clouds and return scaler parameters for inverse operation
    xtr_normalized = bps.normalize(xtr)
    xte_normalized = bps.normalize(xte)

    # this will encode your normalised point clouds with random basis of 512 points,
    # each BPS cell containing l2-distance to closest point
    print("converting data to BPS representation..")
    print("number of basis points: %d" % N_BPS_POINTS)
    print("BPS sampling radius: %f" % BPS_RADIUS)
    print("converting train..")
    xtr_bps = bps.encode(xtr_normalized, n_bps_points=N_BPS_POINTS, bps_cell_type='dists', radius=BPS_RADIUS)
    print("converting test..")
    xte_bps = bps.encode(xte_normalized, n_bps_points=N_BPS_POINTS, bps_cell_type='dists', radius=BPS_RADIUS)

    dataset_tr = pt.utils.data.TensorDataset(pt.Tensor(xtr_bps), pt.Tensor(ytr[:, 0]).long())
    tr_loader = pt.utils.data.DataLoader(dataset_tr, batch_size=512, shuffle=True)

    dataset_te = pt.utils.data.TensorDataset(pt.Tensor(xte_bps), pt.Tensor(yte[:, 0]).long())
    te_loader = pt.utils.data.DataLoader(dataset_te, batch_size=512, shuffle=True)

    n_bps_features = xtr_bps.shape[1]
    n_classes = 40

    print("defining the model..")
    model = ShapeClassifierMLP(n_features=n_bps_features, n_classes=n_classes, hsize1=512, hsize2=512, dropout1=0.8,
                               dropout2=0.6)

    optimizer = pt.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 800
    pbar = range(0, n_epochs)
    test_accs = []
    test_losses = []

    print("training started..")
    model = model.to(DEVICE)

    for epoch_idx in pbar:
        fit(model, DEVICE, tr_loader, optimizer)
        if epoch_idx == 750:
            for param_group in optimizer.param_groups:
                print("decreasing the learning rate to 1e-4..")
                param_group['lr'] = 1e-4
        if epoch_idx % 10 == 0:
            test_loss, test_acc = test(model, DEVICE, te_loader, epoch_idx)
            test_accs.append(test_acc)
            test_losses.append(test_loss)

    _, test_acc = test(model, DEVICE, te_loader, n_epochs)
    print("finished. test accuracy: %f " % test_acc)
    return


if __name__ == '__main__':
    main()