"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from _util import AverageMeter
from _util import binary_accuracy


class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x


def local_a_distance_svm(source_feature, target_feature, source_label, target_label, num_class=31):

    print("Local_a_distance_svm")
    distance = np.zeros(num_class)
    counts = np.zeros(num_class)
    for i in range(num_class):
       
        print("Local_a_distance for classes: {}".format(i))
        source_feature_sub = source_feature[source_label == i]
        target_feature_sub = target_feature[target_label == i]
        
        distance[i], counts[i] = a_distance_svm(source_feature_sub, target_feature_sub, True, True)
    distance_weight = np.sum(distance * (counts / np.sum(counts)))
    distance_avg = np.mean(distance)
    return distance_weight, distance_avg

def a_distance_svm(source_X, target_X, is_from_local=False, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)
    if is_from_local:
        return 2 * (1. - 2 * best_risk), np.shape(tese_X)[0]

    return 2 * (1. - 2 * best_risk)

def local_a_distance(source_feature, target_feature, source_label, target_label, device=None, num_class=31, progress=True, training_epochs=10):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Local_a_distance")
    distance = torch.zeros(num_class)
    counts = torch.zeros(num_class)
    for i in range(num_class):
       
        print("Local_a_distance for classes: {}".format(i))
        source_feature_sub = source_feature[source_label == i]
        target_feature_sub = target_feature[target_label == i]
        
        distance[i], counts[i] = calculate(source_feature_sub, target_feature_sub, device, True, progress, training_epochs)
    distance_weight = torch.sum(distance * (counts / torch.sum(counts)))
    distance_avg = torch.mean(distance)
    return distance_weight, distance_avg

def calculate(source_feature: torch.Tensor, target_feature: torch.Tensor,
              device, is_from_local=False, progress=True, training_epochs=10):
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.

    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier

    Returns:
        :math:`\mathcal{A}`-distance
    """
    source_label = torch.ones((source_feature.shape[0], 1))
    target_label = torch.zeros((target_feature.shape[0], 1))
    feature = torch.cat([source_feature, target_feature], dim=0)
    label = torch.cat([source_label, target_label], dim=0)

    dataset = TensorDataset(feature, label)
    length = len(dataset)
    train_size = int(0.5 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    anet = ANet(feature.shape[1]).to(device)
    optimizer = SGD(anet.parameters(), lr=0.01)
    a_distance = 2.0
    for epoch in range(training_epochs):
        anet.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)
            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()

        anet.eval()
        meter = AverageMeter("accuracy", ":4.2f")
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y = anet(x)
                acc = binary_accuracy(y, label)
                meter.update(acc, x.shape[0])
        error = 1 - meter.avg / 100
        a_distance = 2 * (1 - 2 * error)
        if progress:
            print("epoch {} accuracy: {} A-dist: {}".format(epoch, meter.avg, a_distance))
    if is_from_local:
        return a_distance, val_size

    return a_distance

