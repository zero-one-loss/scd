import time
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
import torch


def evaluation(data_loader, use_cuda, device, dtype, net, key, criterion):
    a = time.time()
    pred = []
    labels = []
    net.eval()
    yps = []
    label = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if use_cuda:
                data, target = data.type_as(net._modules[list(net._modules.keys())[0]].weight), target.to(device=device)

            yp = net(data, layer=net.layers[-1]+'_projection')

            yps.append(yp)
            label.append(target)
            if yp.size(1) == 1:
                outputs = net.signb(yp).round().flatten()
            else:
                outputs = yp.argmax(dim=1)

            pred.append(outputs.cpu().numpy())
            labels.append(target.cpu().numpy())
        yp = torch.cat(yps, dim=0)

        label = torch.cat(label, dim=0)
        loss = criterion(yp, label)
        loss = loss.item()
        pred = np.concatenate(pred, axis=0)
        labels = np.concatenate(labels, axis=0)
        acc = accuracy_score(labels, pred)
        balanced_acc = balanced_accuracy_score(labels, pred)
        print("{} balanced Accuracy: {:.5f}, imbalanced Accuracy: {:.5f}, loss: {:.5f} "
              "cost {:.2f} seconds".format(key, balanced_acc, acc, loss, time.time() - a))

    return acc, loss


def evaluation_text(data_loader, use_cuda, device, dtype, net, key, embedding, criterion):
    a = time.time()
    pred = []
    labels = []
    net.eval()
    yps = []
    label = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            data = batch.text
            target = batch.label
            if use_cuda:
                data, target = data.to(device=device), target.to(device=device)
            data = embedding(data).unsqueeze_(dim=1)
                # data, target = data.type_as(net._modules[list(net._modules.keys())[0]].weight), target.to(device=device)

            yp = net(data, layer=net.layers[-1]+'_projection')
            yps.append(yp)
            label.append(target)

            if yp.size(1) == 1:
                outputs = net.signb(yp).round().flatten()
            else:
                outputs = yp.argmax(dim=1)

            pred.append(outputs.cpu().numpy())
            labels.append(target.cpu().numpy())

    yp = torch.cat(yps, dim=0)
    label = torch.cat(label, dim=0)
    loss = criterion(yp, label)
    loss = loss.item()
    pred = np.concatenate(pred, axis=0)
    labels = np.concatenate(labels, axis=0)
    acc = accuracy_score(labels, pred)
    balanced_acc = balanced_accuracy_score(labels, pred)
    print("{} balanced Accuracy: {:.5f}, imbalanced Accuracy: {:.5f}, loss: {:.5f} "
          "cost {:.2f} seconds".format(key, balanced_acc, acc, loss, time.time() - a))
    return acc

def get_features(data_loader, use_cuda, device, dtype, net, key):
    layers = net.layers
    features = {}
    for layer in layers:
        features[layer] = []

    for batch_idx, (data, target) in enumerate(data_loader):
        if use_cuda:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        for layer in layers:
            features[layer].append(net(data, layer=layer+'_projection').sign().cpu())

    for layer in layers:
        features[layer] = torch.cat(features[layer], dim=0)

    return features