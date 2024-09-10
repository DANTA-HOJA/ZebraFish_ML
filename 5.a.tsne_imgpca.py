# %%
import os
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
# import igraph
import rich
from PIL import Image, ImageOps
from rich.pretty import Pretty
from scipy.spatial import distance
from skimage import io
from sklearn.decomposition import PCA
from sklearn.feature_extraction import image
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.shared.config import load_config
from modules.shared.utils import create_new_dir, get_repo_root

# -----------------------------------------------------------------------------/
# %%
# notebook name
notebook_name = Path(__file__).stem

# -----------------------------------------------------------------------------/
# %%
# load config
config = load_config("ml_analysis.toml")
palmskin_result_name: Path = Path(config["data_processed"]["palmskin_result_name"])
cluster_desc: str = config["data_processed"]["cluster_desc"]
dark: int = config["SLIC"]["dark"]
img_mode: str = config["ML"]["img_mode"]
img_resize: tuple = tuple(config["ML"]["img_resize"])
rich.print("", Pretty(config, expand_all=True))

# -----------------------------------------------------------------------------/
# %%
processed_di = ProcessedDataInstance()
processed_di.parse_config("ml_analysis.toml")

# src
repo_root = get_repo_root()
slic_dirname = f"{palmskin_result_name.stem}_{{dark_{dark}}}"
ml_csv = repo_root.joinpath("data/generated/ML", processed_di.instance_name,
                            cluster_desc, slic_dirname, "ml_dataset.csv")

# dst
dst_dir = ml_csv.parents[1].joinpath(f"{palmskin_result_name.stem}.imgtsne")
create_new_dir(dst_dir)

# -----------------------------------------------------------------------------/
# %%
df = pd.read_csv(ml_csv, encoding='utf_8_sig', index_col="palmskin_dname")
print(f"Read ML Dataset: '{ml_csv}'")
df

# -----------------------------------------------------------------------------/
# %% [markdown]
# ## Training

# Parse all images in custom folder, saves their color and path into the data array
# See: `save_wikidata_images.ipynb` to create a folder containing images from Wikidata.

# -----------------------------------------------------------------------------/
# %%
rel_path, _ = processed_di.get_sorted_results_dict("palmskin", palmskin_result_name)
print(rel_path)

data = []
for palmskin_dname in tqdm(df.index):
    img_path: Path = processed_di.palmskin_processed_dir.joinpath(palmskin_dname, rel_path)
    image = cv2.imread(str(img_path))
    if image is not None:
        image = cv2.cvtColor(image, getattr(cv2, f"COLOR_BGR2{img_mode}"))
        image = cv2.resize(image, img_resize, interpolation=cv2.INTER_CUBIC)
        image = image.flatten()
        data.append([image, img_path])

features, img_paths = zip(*data)
features: list[np.ndarray]
img_paths: list[Path]

# -----------------------------------------------------------------------------/
# %% [markdown]
# images instantiate a PCA object, which we will then fit our data to,
# choosing to keep the top `n` principal components. This may take a few minutes.

# -----------------------------------------------------------------------------/
# %%
rand_seed = int(cluster_desc.split("_")[-1].replace("RND", ""))

features = np.array(features) # shape = (n, 45*45*3)
pca = PCA(n_components=5, random_state=rand_seed)
pca.fit(features)
pca_features = pca.transform(features)

# -----------------------------------------------------------------------------/
# %%
num_images_to_plot = len(img_paths) # if specify a number, only random select "n" samples to create the map

if len(img_paths) > num_images_to_plot:
    sort_order = sorted(random.sample(range(len(img_paths)), num_images_to_plot))
    img_paths = [img_paths[i] for i in sort_order]
    pca_features = [pca_features[i] for i in sort_order]

# -----------------------------------------------------------------------------/
# %%
X = np.array(pca_features)

# 特徵標準化處理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(n_components=2, perplexity=20, n_iter=50000,
            random_state=rand_seed, verbose=2).fit_transform(X_scaled)

# -----------------------------------------------------------------------------/
# %% [markdown]
# Internally, t-SNE uses an iterative approach, making small (or sometimes large) adjustments to the points. By default, t-SNE will go a maximum of 1000 iterations, but in practice, it often terminates early because it has found a locally optimal (good enough) embedding.
# The variable t-SNE contains an array of unnormalized 2d points, corresponding to the embedding. In the next cell, we normalize the embedding so that lies entirely in the range (0,1).

# -----------------------------------------------------------------------------/
# %% [markdown]
# ## Plots the clusters

# -----------------------------------------------------------------------------/
# %%
def add_colored_border(image, border_color, border_size):
    """
    """
    # 轉換 RGB 色碼
    border_color_rgb = (border_color[0], border_color[1], border_color[2])

    # 新增外框
    bordered_image = ImageOps.expand(image, border=border_size, fill=border_color_rgb)

    # 顯示加上外框的圖像
    return bordered_image

# -----------------------------------------------------------------------------/
# %%
tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

# -----------------------------------------------------------------------------/
# %%
import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

width = 12000
height = 9000
max_dim = 400

# image border
border_size = 20
cls2color = {
    "S": (0, 255, 0), # green
    "M": (0, 0, 255), # blue
    "L": (255, 0, 0) # red
}

full_image = Image.new('RGBA', (width, height))
for path, palmskin_dname, x, y in zip(img_paths, df.index, tx, ty):
    # read image
    image = Image.open(path)
    
    # add border according to `fish_class`
    fish_class = df.loc[palmskin_dname, "class"]
    tile = add_colored_border(image, cls2color[fish_class], border_size)
    
    rs = max(1, tile.width/max_dim, tile.height/max_dim)
    tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.Resampling.LANCZOS)
    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=300)
ax.set_title(f"t-SNE ({img_mode} image, first 5 PCA components)")

# set legend
for k, v in cls2color.items():
    ax.plot([], [], color=np.array(v)/255, label=k)
ax.legend()

# remove ticks
ax.set_xticks([])
ax.set_yticks([])

ax.imshow(full_image)
fig.tight_layout()

# -----------------------------------------------------------------------------/
# %% [markdown]
# ### Save `t-SNE` results

# -----------------------------------------------------------------------------/
# %%
import json

fig.savefig(dst_dir.joinpath(f"tSNE.{img_mode}.png"))

data = [{"path": str(path.resolve()), "point": [float(x), float(y)]}
            for path, x, y in zip(img_paths, tx, ty)]

tsne_pt_path = dst_dir.joinpath(f"tSNE.{img_mode}.json")
with open(tsne_pt_path, 'w') as f_writer:
    json.dump(data, f_writer, indent=4)

print(f"Save t-SNE results to : '{tsne_pt_path.parent}'")

# -----------------------------------------------------------------------------/
# %% [markdown]
# ### Paste into grid using [RasterFairy-Py3](https://github.com/Quasimondo/RasterFairy.git)

# -----------------------------------------------------------------------------/
# %%
import rasterfairy

grid_assignment = rasterfairy.transformPointCloud2D(tsne)

gpx, gpy = zip(*grid_assignment[0])
print(np.max(gpx), np.max(gpy), grid_assignment[1])

# -----------------------------------------------------------------------------/
# %%
nx = grid_assignment[1][0]
ny = grid_assignment[1][1]

tile_width = img_resize[0]
tile_height = img_resize[1]

full_width = tile_width * nx
full_height = tile_height * ny
aspect_ratio = float(tile_width) / tile_height

grid_image = Image.new('RGBA', (full_width, full_height))

for path, palmskin_dname, grid_pos in zip(img_paths, df.index, grid_assignment[0]):
    # read image
    image = Image.open(path)
    
    # add border according to `fish_class`
    fish_class = df.loc[palmskin_dname, "class"]
    tile = add_colored_border(image, cls2color[fish_class], border_size)
    
    # get x, y
    idx_x, idx_y = grid_pos
    x, y = tile_width * idx_x, tile_height * idx_y
    
    tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
    if (tile_ar > aspect_ratio):
        margin = 0.5 * (tile.width - aspect_ratio * tile.height)
        tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
    else:
        margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
        tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
    tile = tile.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
    grid_image.paste(tile, (int(x), int(y)))

# -----------------------------------------------------------------------------/
# %% [markdown]
# ### Save the grid image

# -----------------------------------------------------------------------------/
# %%
background = Image.new("RGB", grid_image.size, (255, 255, 255))
background.paste(grid_image, mask=grid_image.split()[3]) # 3 is the alpha channel
background.save(dst_dir.joinpath(f"grid-tSNE.{img_mode}.png"))