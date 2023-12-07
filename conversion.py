# %% [markdown]
# ENVIRON

# %%
from math import ceil
import os
import sys
import time
from datetime import datetime
sys.path.insert(0, '/home/work/jupyter/dongjae/2023-fall/Lecture/Graph/GNN-Superpixel/boruvka-superpixel/pybuild')

# %%
from tqdm import tqdm, trange
from pandarallel import pandarallel
import swifter
from swifter import set_defaults
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# %%
from PIL import Image
from PIL import ImageFilter
import skimage as ski
from skimage.segmentation import slic
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
from rdp import rdp
import boruvka_superpixel

# %%
import networkx as nx

# %%
from IPython.display import clear_output
import ipywidgets

# %%
from itertools import combinations

# %%
tqdm.pandas()
pandarallel.initialize(
    progress_bar = False,       # True,
    nb_workers = 8,
    verbose = 2,
    use_memory_fs = None,
)
swifter.set_defaults(
    npartitions = 8,
    progress_bar = True,
    progress_bar_desc = 'Working...'
)
UNIT = 1000
NUM_SUPERPIXEL = 75

# %% [markdown]
# DATASET

# %%
filenames = []
labels = []

# %%
for foldername in tqdm(os.listdir(f'/home/work/jupyter/dongjae/2023-fall/Lecture/Graph/GNN-Superpixel/ImageNet/train')):
    for filename in os.listdir(f'/home/work/jupyter/dongjae/2023-fall/Lecture/Graph/GNN-Superpixel/ImageNet/train/{foldername}'):
        filenames.append(f'/home/work/jupyter/dongjae/2023-fall/Lecture/Graph/GNN-Superpixel/ImageNet/train/{foldername}/{filename}')
        labels.append(foldername)

# %%
df = pd.DataFrame(columns = ['filename', 'label'])

# %%
num_processed = len(os.listdir(f'/home/work/jupyter/dongjae/2023-fall/Lecture/Graph/GNN-Superpixel/converted_{NUM_SUPERPIXEL}/train'))

# %% [markdown]
# CONVERSION UNDER CHUNKING

# %%
def rgb2lab(image):
    return ski.color.rgb2lab(image)

# %%
def get_slic(img, num):
    return slic(img, num) - 1

# %%
def get_boruvka(img, num):
    img_edge = np.zeros((img.shape[:2]), dtype=img.dtype)
    bosupix = boruvka_superpixel.BoruvkaSuperpixel()
    bosupix.build_2d(img, img_edge)
    return bosupix.label(num)

# %%
def get_networkx(image, format, algo):
    assert format in ['rgb', 'lab']
    assert algo in ['slic', 'boruvka']
    
    if image[f'{algo}_{format}'].max() == 0:
        return nx.Graph()

    return nx.Graph(ski.graph.rag_mean_color(image[format], image[f'{algo}_{format}']))

# %%
def get_approxed_polygon(label_img):
    polygons = []
    
    for index in range(label_img.max()):
        mask = Image.fromarray((label_img == index).astype(np.uint8) * 255)
        contour = mask.filter(ImageFilter.CONTOUR)
        points = find_contours((label_img == index).astype(np.uint8), 0)[0]
        points = subdivide_polygon(points, degree = 2, preserve_ends = True)
        points = approximate_polygon(points, tolerance = 8)
        polygons.append(points)\
    
    return polygons

# %%
def get_local_graphs(coordinates):
    graphs = []

    for coordinate in coordinates:
        mean = coordinate.mean(axis = 0)
        G = nx.Graph()
        for index, coor in enumerate(coordinate):
            coor = coor - mean
            G.add_node(index, coor = coor)
        for index in range(G.number_of_nodes()):
            u = G.nodes()[index]
            v = G.nodes()[(index + 1) % G.number_of_nodes()]
            G.add_edge(index, (index + 1) % G.number_of_nodes(), weight = np.linalg.norm(u['coor'] - v['coor']))
        graphs.append(G)

    return graphs

# %%
for chunk_id in trange(ceil(len(filenames) / UNIT)):
    if chunk_id < num_processed:
        continue

    df = pd.DataFrame(columns = ['filename', 'label'])
    chunk_files = filenames[chunk_id * UNIT:(chunk_id + 1) * UNIT]
    chunk_labels = labels[chunk_id * UNIT:(chunk_id + 1) * UNIT]
    
    df['filename'] = chunk_files
    df['label'] = chunk_labels
    
    # RGB and LAB Image
    df['rgb'] = df['filename'].parallel_apply(lambda x : np.array(Image.open(x).convert('RGB').resize((128, 128))))
    df['lab'] = df['rgb'].parallel_apply(ski.color.rgb2lab)
    
    # SLIC and BORUVKA superpixels
    df['slic_rgb'] = df['rgb'].parallel_apply(lambda x : get_slic(x, NUM_SUPERPIXEL))
    df['slic_lab'] = df['lab'].parallel_apply(lambda x : get_slic(x, NUM_SUPERPIXEL))
    df['boruvka_rgb'] = df['rgb'].parallel_apply(lambda x : get_boruvka(x, NUM_SUPERPIXEL))
    df['boruvka_lab'] = df['lab'].parallel_apply(lambda x : get_boruvka(x, NUM_SUPERPIXEL))
    
    # Num of superpixels
    df['slic_rgb_count'] = df['slic_rgb'].parallel_apply(lambda x : x.max())
    df['slic_lab_count'] = df['slic_lab'].parallel_apply(lambda x : x.max())
    df['boruvka_rgb_count'] = df['boruvka_rgb'].parallel_apply(lambda x : x.max())
    df['boruvka_lab_count'] = df['boruvka_lab'].parallel_apply(lambda x : x.max())

    # Global Graphs
    df['slic_rgb_global_graph'] = df.parallel_apply(lambda x : get_networkx(x, 'rgb', 'slic'), axis = 1)
    df['slic_lab_global_graph'] = df.parallel_apply(lambda x : get_networkx(x, 'lab', 'slic'), axis = 1)
    df['boruvka_rgb_global_graph'] = df.parallel_apply(lambda x : get_networkx(x, 'rgb', 'boruvka'), axis = 1)
    df['boruvka_lab_global_graph'] = df.parallel_apply(lambda x : get_networkx(x, 'lab', 'boruvka'), axis = 1)
    
    # Local coordinates
    df['slic_rgb_local_coor'] = df['slic_rgb'].parallel_apply(get_approxed_polygon)
    df['slic_lab_local_coor'] = df['slic_lab'].parallel_apply(get_approxed_polygon)
    df['boruvka_rgb_local_coor'] = df['boruvka_rgb'].parallel_apply(get_approxed_polygon)
    df['boruvka_lab_local_coor'] = df['boruvka_lab'].parallel_apply(get_approxed_polygon)
    
    # Local Graphs
    df['slic_rgb_local_graph'] = df['slic_rgb_local_coor'].parallel_apply(get_local_graphs)
    df['slic_lab_local_graph'] = df['slic_lab_local_coor'].parallel_apply(get_local_graphs)
    df['boruvka_rgb_local_graph'] = df['boruvka_rgb_local_coor'].parallel_apply(get_local_graphs)
    df['boruvka_lab_local_graph'] = df['boruvka_lab_local_coor'].parallel_apply(get_local_graphs)

    df.to_pickle(f'/home/work/jupyter/dongjae/2023-fall/Lecture/Graph/GNN-Superpixel/converted_{NUM_SUPERPIXEL}/train/data{chunk_id:04d}.pkl')

    break

# %%



