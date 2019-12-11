import numpy as np
import os
import sys
from functools import partial
import multiprocessing

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
BPS_CACHE_FILE = os.path.join(DATA_PATH, 'bps_conv3d_data.npz')

N_BPS_POINTS = 32**3
BPS_RADIUS = 1.2
DEVICE = sys.argv[1]  # 'cpu' or 'cuda'


class ShapeClassifierConv3D(nn.Module):

    def __init__(self, n_features, n_classes):
        super(ShapeClassifierConv3D, self).__init__()

        self.conv11 = nn.Conv3d(in_channels=n_features, out_channels=8, kernel_size=(3, 3, 3))
        self.bn11 = nn.BatchNorm3d(8)
        self.conv12 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3))
        self.bn12 = nn.BatchNorm3d(16)
        self.mp1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv21 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))
        self.bn21 = nn.BatchNorm3d(32)
        self.conv22 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3))
        self.bn22 = nn.BatchNorm3d(64)
        self.mp2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.do1 = nn.Dropout(0.8)
        self.fc1 = nn.Linear(in_features=8000, out_features=2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.do2 = nn.Dropout(0.8)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.bn2 = nn.BatchNorm1d(512)
        self.do3 = nn.Dropout(0.8)
        self.fc3 = nn.Linear(in_features=512, out_features=n_classes)

    def forward(self, x):
        x = self.bn11(F.relu(self.conv11(x)))
        x = self.bn12(F.relu(self.conv12(x)))
        x = self.mp1(x)

        x = self.bn21(F.relu(self.conv21(x)))
        x = self.bn22(F.relu(self.conv22(x)))
        x = self.mp2(x)

        x = self.do1(x.reshape([-1, 8000]))
        x = self.do1(self.bn1(F.relu(self.fc1(x))))
        x = self.do3(self.bn2(F.relu(self.fc2(x))))

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

    if not os.path.exists(BPS_CACHE_FILE):
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

        n_cpus = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(n_cpus)
        bps_encode_func = partial(bps.encode, bps_arrangement='grid', n_bps_points=32 ** 3, radius=1.2,
                                  bps_cell_type='dists')

        # xtr_bps = bps.encode(xtr_normalized, bps_arrangement='grid', n_bps_points=N_BPS_POINTS, radius=BPS_RADIUS,
        #                      bps_cell_type='deltas')
        xtr_bps = np.concatenate(pool.map(bps_encode_func, np.array_split(xtr, n_cpus)), 0)

        xtr_bps = xtr_bps.reshape([-1, 32, 32, 32, 3])

        print("converting test..")
        xte_bps = np.concatenate(pool.map(bps_encode_func, np.array_split(xte, n_cpus)), 0)

        # xte_bps = bps.encode(xte_normalized, bps_arrangement='grid', n_bps_points=N_BPS_POINTS, radius=BPS_RADIUS,
        #                      bps_cell_type='deltas')
        xte_bps = xte_bps.reshape([-1, 32, 32, 32, 3])
        print("saving cache file for future runs..")
        np.savez(BPS_CACHE_FILE, xtr=xtr_bps, ytr=ytr, xte=xte_bps, yte=yte)
    else:
        print("loading converted data from cache..")
        data = np.load(BPS_CACHE_FILE)
        xtr_bps = data['xtr']
        ytr = data['ytr']
        xte_bps = data['xte']
        yte = data['yte']

    xtr_bps = xtr_bps.transpose(0, 4, 2, 3, 1)
    dataset_tr = pt.utils.data.TensorDataset(pt.Tensor(xtr_bps), pt.Tensor(ytr[:, 0]).long())
    tr_loader = pt.utils.data.DataLoader(dataset_tr, batch_size=512, shuffle=True)

    xte_bps = xte_bps.transpose(0, 4, 2, 3, 1)
    dataset_te = pt.utils.data.TensorDataset(pt.Tensor(xte_bps), pt.Tensor(yte[:, 0]).long())
    te_loader = pt.utils.data.DataLoader(dataset_te, batch_size=512, shuffle=True)

    n_bps_features = xtr_bps.shape[1]
    n_classes = 40

    print("defining the model..")
    model = ShapeClassifierConv3D(n_features=n_bps_features, n_classes=n_classes)

    optimizer = pt.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 200
    pbar = range(0, n_epochs)
    test_accs = []
    test_losses = []

    print("training started..")
    model = model.to(DEVICE)

    for epoch_idx in pbar:
        fit(model, DEVICE, tr_loader, optimizer)
        if epoch_idx == 50:
            for param_group in optimizer.param_groups:
                print("decreasing the learning rate to 1e-4..")
                param_group['lr'] = 1e-4
        if epoch_idx == 150:
            for param_group in optimizer.param_groups:
                print("decreasing the learning rate to 1e-5..")
                param_group['lr'] = 1e-5
        if epoch_idx % 1 == 0:
            test_loss, test_acc = test(model, DEVICE, te_loader, epoch_idx)
            test_accs.append(test_acc)
            test_losses.append(test_loss)

    _, test_acc = test(model, DEVICE, te_loader, n_epochs)
    print("finished. test accuracy: %f " % test_acc)
    return


if __name__ == '__main__':
    main()