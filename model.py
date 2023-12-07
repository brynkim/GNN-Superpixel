# %% [markdown]
# PACKAGE

# %%
from tqdm import tqdm, trange
import argparse
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
from torch_geometric.nn import Sequential, GATConv, global_mean_pool
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx

# %%
import numpy as np
import pandas as pd

# %%
import networkx as nx

# %%
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# %% [markdown]
# CONFIG

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
tqdm.pandas()

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'Cora')
parser.add_argument('--hidden_channels', type = int, default = 8)
parser.add_argument('--heads', type = int, default = 8)
parser.add_argument('--lr', type = float, default = 0.005)
parser.add_argument('--weight_decay', type = float, default = 0.0)
parser.add_argument('--epochs', type = int, default = 10)
parser.add_argument('--wandb', action = 'store_true', help = 'Track experiment')
parser.add_argument('--superpixel', type = str, default = 'slic')
parser.add_argument('--image', type = str, default = 'rgb')
parser.add_argument('--batch_size', type = int, default = 16)
args = parser.parse_known_args()[0]

# %% [markdown]
# DATASET

# %%
selected_df = pd.read_csv('./select.csv', delimiter = ',')
selected_df['id'] = range(len(selected_df))
selected_df = selected_df.loc[:4]                 # HACK: Limit the number for classification

# %%
class2id = {directory: id for directory, id in zip(selected_df['directory'], selected_df['id'])}

# %%
print('Import train data')
dfs = []

for filename in tqdm(os.listdir('./filtered/train')):
    df = pd.read_pickle(f'./filtered/train/{filename}')
    df = df[df['label'].apply(lambda x : x in class2id.keys())]
    df['label'] = df['label'].apply(lambda x : class2id[x])
    df = df[[f'{args.superpixel}_{args.image}_global_graph', f'{args.superpixel}_{args.image}', f'{args.image}', 'label']]
    df.columns = ['graph', 'superpixel', 'image', 'label']
    dfs.append(df)

train_df = pd.concat(dfs).reset_index(drop = True)

# %%
print('Import validation data')
dfs = []

for filename in tqdm(os.listdir('./filtered/val')):
    df = pd.read_pickle(f'./filtered/val/{filename}')
    df = df[df['label'].apply(lambda x : x in class2id.keys())]
    df['label'] = df['label'].apply(lambda x : class2id[x])
    df = df[[f'{args.superpixel}_{args.image}_global_graph', f'{args.superpixel}_{args.image}', f'{args.image}', 'label']]
    df.columns = ['graph', 'superpixel', 'image', 'label']
    dfs.append(df)

valid_df = pd.concat(dfs).reset_index(drop = True)

# %%
num_classes = valid_df['label'].max() + 1
num_classes

# %% [markdown]
# FEATURE ENGINEERING

# %%
def get_supix_statistics(data):
    graph = data['graph']
    superpixel = data['superpixel']
    image = data['image']
    label = data['label']
    num_superpixel = data['superpixel'].max()

    means = []
    stds = []
    centroids = []

    for supix in range(superpixel.max() + 1):
        mask = superpixel != supix      # Mask out the pixels that are not equal to given superpixel label.
        trinary_mask = np.stack([mask, mask, mask], axis = 2)

        masked = np.ma.masked_array(image, trinary_mask)
        mean = np.ma.mean(masked, axis = (0, 1))
        std = np.ma.std(masked, axis = (0, 1))
        centroid = np.array([np.mean(subset) for subset in np.nonzero(np.logical_not(mask))])

        means.append(mean.data)
        stds.append(std.data)
        centroids.append(centroid)

    return means, stds, centroids

# %%
print('Compute statistics for train data')
ret = train_df.progress_apply(get_supix_statistics, axis = 1)
train_df['means'] = [r[0] for r in ret]
train_df['stds'] = [r[1] for r in ret]
train_df['centroids'] = [r[2] for r in ret]

# %%
print('Compute statistics for validation data')
ret = valid_df.progress_apply(get_supix_statistics, axis = 1)
valid_df['means'] = [r[0] for r in ret]
valid_df['stds'] = [r[1] for r in ret]
valid_df['centroids'] = [r[2] for r in ret]

# %% [markdown]
# CONSTRUCT NODE FEATURES

# %%
visual_model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained = True).to(device)

# %%
def construct_convolution_features(data):
    graph = data['graph']
    superpixel = data['superpixel']
    image = data['image']
    label = data['label']
    num_superpixels = graph.number_of_nodes()
    
    masks = np.stack([superpixel == i for i in range(num_superpixels)])
    masked_images = np.stack([image * np.expand_dims(mask, axis = -1) for mask in masks])
    masked_images.shape
    masked_images = masked_images.transpose(0, 3, 1, 2)
    masked_images = masked_images / 255
    masked_images = torch.tensor(masked_images).float().to(device)
    
    return visual_model(masked_images).detach().cpu()

# %%
def construct_node_features(data):
    graph = data['graph']
    means = data['means']
    stds = data['stds']
    centroids = data['centroids']
    conv_features = data['conv_features']
    weights = dict()

    for index, (mean, std, centroid, conv_feature) in enumerate(zip(means, stds, centroids, conv_features)):
        weight = np.concatenate([mean, std, centroid, conv_feature])
        weights[index] = weight
    
    return weights

# %%
def construct_new_graph(data):
    graph = data['graph']
    attribute = data['attributes']
    for node in graph.nodes():
        graph.nodes[node].clear()
    nx.set_node_attributes(graph, attribute, name = 'features')
    return graph

# %%
def attach_label(data):
    G = data['graph']
    label = data['label']

    D = from_networkx(G, group_node_attrs = ['features'])
    D['y'] = torch.tensor(label)

    return D

# %%
print('Filter graphs')
train_df = train_df[train_df['graph'].progress_apply(lambda x : x.number_of_nodes() != 0)]
valid_df = valid_df[valid_df['graph'].progress_apply(lambda x : x.number_of_nodes() != 0)]

# %%
print('Construct convolutional features')
train_df['conv_features'] = train_df.progress_apply(construct_convolution_features, axis = 1)
valid_df['conv_features'] = valid_df.progress_apply(construct_convolution_features, axis = 1)

# %%
print('Construct node features')
train_df['attributes'] = train_df.progress_apply(construct_node_features, axis = 1)
valid_df['attributes'] = valid_df.progress_apply(construct_node_features, axis = 1)

# %%
print('Construct graphs')
train_df['graph'] = train_df.progress_apply(construct_new_graph, axis = 1)
valid_df['graph'] = valid_df.progress_apply(construct_new_graph, axis = 1)

# %%
print('Build tensor data')
train_df['data'] = train_df.progress_apply(attach_label, axis = 1)
valid_df['data'] = valid_df.progress_apply(attach_label, axis = 1)

# %% [markdown]
# MODEL

# %%
class ImageNetDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(ImageNetDataset, self).__init__()
        self.data_list = data_list
        self.data, self.slices = self.collate(data_list)

# %%
class SPINCS(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(SPINCS, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout = 0.5)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads, dropout = 0.5)
        self.linear = nn.Linear(hidden_channels * heads, out_channels)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.linear(x)
        x = self.softmax(x)

        return x

# %%
train_dataset = ImageNetDataset(train_df['data'].to_list())
valid_dataset = ImageNetDataset(valid_df['data'].to_list())

# %%
train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle = True)

# %%
model = SPINCS(18, args.hidden_channels, num_classes, args.heads).to(device)

# %%
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

# %%
criterion = nn.CrossEntropyLoss()

# %% [markdown]
# TRAIN

# %%
train_losses = []
valid_losses = []
train_accs = []
valid_accs = []
models = []

# %%
print('Train start!')
for epoch in range(args.epochs):
    # Train Mode
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_correct = 0
    train_size = 0

    for data in train_dataloader:
        x = data.x.float().to(device)
        edge_index = data.edge_index.long().to(device)
        batch = data.batch.long().to(device)

        optimizer.zero_grad()
        output = model(x, edge_index, batch)
        pred = output.argmax(axis = 1).cpu()

        loss = criterion(output, F.one_hot(data.y, num_classes = num_classes).float().to(device))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (pred == data.y).sum()
        train_size += data.y.shape[0]

    train_loss = train_loss / train_size
    train_losses.append(train_loss)
    train_acc = train_correct / train_size
    train_accs.append(train_acc)

    # Eval Mode
    model.eval()
    valid_loss = 0.0
    valid_acc = 0.0
    valid_correct = 0
    valid_size = 0

    with torch.no_grad():
        for data in valid_dataloader:
            x = data.x.float().to(device)
            edge_index = data.edge_index.long().to(device)
            batch = data.batch.long().to(device)

            output = model(x, edge_index, batch)
            pred = output.argmax(axis = 1).cpu()

            loss = criterion(output, F.one_hot(data.y, num_classes = num_classes).float().to(device))

            valid_loss += loss.item()
            valid_correct += (pred == data.y).sum()
            valid_size += data.y.shape[0]

        valid_loss = valid_loss / valid_size
        valid_losses.append(valid_loss)
        valid_acc = valid_correct / valid_size
        valid_accs.append(valid_acc)

    models.append(model)
    print(f'Epoch {epoch} finished: train loss - {train_loss}, train acc - {train_acc} / valid loss - {valid_loss}, valid acc - {valid_acc}')

# %%
with open('./models.pkl', 'w') as f:
    pickle.dump(models, f)

# %%
with open('./train_losses.pkl', 'w') as f:
    pickle.dump(train_losses, f)

# %%
with open('./valid_losses.pkl', 'w') as f:
    pickle.dump(valid_losses, f)

# %%



