# %% [markdown]
# ENVIRON

# %%
import os
import sys
from datetime import datetime
sys.path.insert(0, './boruvka/pybuild')

# %%
from tqdm import tqdm, trange
from pandarallel import pandarallel
import swifter
from swifter import set_defaults
import skimage as ski
import numpy as np
import pandas as pd
from PIL import Image
from skimage import graph

# %%
import boruvka_superpixel
from wavelet import utils_wavelet as wv
from skimage.segmentation import slic, felzenszwalb

# %%
import networkx as nx

# %%
tqdm.pandas()
set_defaults(
    npartition = 8,
    dask_threshold = 1,
    scheduler = 'processes',
    progress_bar = True,
    progress_bar_desc = 'Converting...',
    allow_dasks_on_string = False,
    force_parallel = False
)

# %% [markdown]
# DATASET

# %%
images = []

# %%
for filename in os.listdir(f'./ImageNet_ILSVEC_2010_val'):
    images.append(f'./ImageNet_ILSVEC_2010_val/{filename}')

# %%
df = pd.DataFrame(columns = ['filename', 'image', 'label'])

# %%
with open('./ILSVRC2010_validation_ground_truth.txt') as f:
    labels = f.readlines()
labels = [int(label.strip()) for label in labels]

# %%
print('BEFORE IMPORT:', str(datetime.now()))

# %%
for index, filename in enumerate(tqdm(images)):
    df.loc[index] = {'filename': filename, 'image': np.array(Image.open(filename).convert('RGB')), 'label': labels[index]}

# %%
print('AFTER IMPORT:', str(datetime.now()))

# %% [markdown]
# SLIC - about 3 hours per 50k

# %%
def slic_conversion(image, num_sp):
    try:
        return slic(image, num_sp)
    except:
        return np.array([[]])

# %%
print('BEFORE SLIC LABEL:', str(datetime.now()))

# %%
df['slic_label'] = df['image'].swifter.progress_bar(True).apply(lambda x : slic_conversion(x, 75))

# %%
print('AFTER SLIC LABEL:', str(datetime.now()))

# %%
print('BEFORE SLIC GRAPH:', str(datetime.now()))

# %%
df['slic_graph'] = df.swifter.apply(lambda x : graph.rag_mean_color(x['image'], x['slic_label']), axis = 1)

# %%
print('AFTER SLIC GRAPH:', str(datetime.now()))

# %% [markdown]
# BORUVKA SUPERPIXEL HIERARCHY - about 1 hour per 50k

# %%
def boruvka_conversion(image, num_sp):
    try:
        image_edge = np.zeros((image.shape[:2]), dtype = image.dtype)

        bsp = boruvka_superpixel.BoruvkaSuperpixel()
        bsp.build_2d(image, image_edge)
        bsp_label = bsp.label(num_sp)

        return bsp_label
    except:
        pass

# %%
print('BEFORE BORUVKA LABEL:', str(datetime.now()))

# %%
df['boruvka_label'] = df['image'].parallel_apply(lambda x : boruvka_conversion(x, 75))

# %%
print('BEFORE BORUVKA LABEL:', str(datetime.now()))

# %%
print('BEFORE BORUVKA GRAPH:', str(datetime.now()))

# %%
df['boruvka_graph'] = df.parallel_apply(lambda x : graph.rag_mean_color(df['image'], df['boruvka_label']), axis = 1)

# %%
print('AFTER BORUVKA GRAPH:', str(datetime.now()))

# %% [markdown]
# SAVE FOR TEMP

# %%
df.to_csv('./imagenet50k.csv')


