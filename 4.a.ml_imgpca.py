# %%
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rich
from rich.pretty import Pretty
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from modules.data import dname
from modules.data.processeddatainstance import ProcessedDataInstance
from modules.dl.tester.utils import confusion_matrix_with_class
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
dst_dir = ml_csv.parents[1].joinpath(f"{palmskin_result_name.stem}.imgpca")
create_new_dir(dst_dir)

# -----------------------------------------------------------------------------/
# %%
df = pd.read_csv(ml_csv, encoding='utf_8_sig')
print(f"Read ML Dataset: '{ml_csv}'")
df

# -----------------------------------------------------------------------------/
# %%
labels = sorted(Counter(df["class"]).keys())
label2idx = {label: idx for idx, label in enumerate(labels)}
rich.print(f"labels = {labels}")
rich.print(f"label2idx = {label2idx}")

# -----------------------------------------------------------------------------/
# %%
training_df = df[(df["dataset"] == "train") | (df["dataset"] == "valid")]
test_df = df[(df["dataset"] == "test")]

training_df

# -----------------------------------------------------------------------------/
# %% [markdown]
# ## Training

# -----------------------------------------------------------------------------/
# %%
rel_path, _ = processed_di.get_sorted_results_dict("palmskin", str(palmskin_result_name))
print(rel_path)

data = []
for palmskin_dname in tqdm(training_df["palmskin_dname"]):
    img_path: Path = processed_di.palmskin_processed_dir.joinpath(palmskin_dname, rel_path)
    image = cv2.imread(str(img_path))
    if image is not None:
        image = cv2.cvtColor(image, getattr(cv2, f"COLOR_BGR2{img_mode}"))
        image = cv2.resize(image, img_resize, interpolation=cv2.INTER_CUBIC)
        image = image.flatten()
        data.append([image, img_path])

features, img_paths  = zip(*data)

# -----------------------------------------------------------------------------/
# %%
rand_seed = int(cluster_desc.split("_")[-1].replace("RND", ""))

# image pca
features = np.array(features) # convert type, shape = (n, 45*45*3)
pca = PCA(n_components=5, random_state=rand_seed)
pca.fit(features)
input_training = pca.transform(features)

# 初始化 Random Forest 分類器
random_forest = RandomForestClassifier(n_estimators=100, random_state=rand_seed)

# 訓練模型
idx_gt_training = [label2idx[c_label] for c_label in training_df["class"]]
random_forest.fit(input_training, idx_gt_training)

# -----------------------------------------------------------------------------/
# %%
import matplotlib.pyplot as plt

prop_var = pca.explained_variance_ratio_
eigenvalues = pca.explained_variance_
 
plt.plot(np.arange(pca.n_components_), 
         prop_var, 
         'ro-')
plt.title('Figure 1: Scree Plot', fontsize=8)
plt.ylabel('Proportion of Variance', fontsize=8)
plt.show()

accum_prop_var = 0
for i, prop_var in enumerate(pca.explained_variance_ratio_):
    accum_prop_var += prop_var
    if accum_prop_var > 0.9:
        print("{} PCA components, accum_prop_var: {}".format(i, accum_prop_var))
        break

print("{} PCA components, total_prop_var: {}".format(len(pca.explained_variance_ratio_),
                                                     np.sum(pca.explained_variance_ratio_)))
print("PCA feature_importances: {}".format(random_forest.feature_importances_))

# -----------------------------------------------------------------------------/
# %%
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 3})
fig, axes = plt.subplots(1, 5, figsize=(4, 3), dpi=1000)
assert len(axes.flatten()) == len(pca.components_)

for i, ax in enumerate(axes.flatten()):
    pc = pca.components_[i].reshape(*img_resize[::-1], 3)
    pc = (pc-np.min(pc)) / (np.max(pc) - np.min(pc))
    ax.imshow(pc)
    ax.set_title("pc{} ({:.4f})".format(i, random_forest.feature_importances_[i]))

fig.tight_layout()

# -----------------------------------------------------------------------------/
# %%
plt.rcParams.update({'font.size': 3})
fig2, axes2 = plt.subplots(1, 2, figsize=(4,3), dpi=1000)

# normal
pc = np.mean((pca.components_), axis=0)
pc = pc.reshape(*img_resize[::-1], 3)
pc = (pc-np.min(pc)) / (np.max(pc) - np.min(pc))
axes2[0].imshow(pc)

# weighted
from copy import deepcopy

pca_weighted_c = deepcopy(pca.components_)
for i, (pc, fimp) in enumerate(zip(pca_weighted_c, random_forest.feature_importances_)):
    pca_weighted_c[i] = pc*fimp

pc = np.mean(pca_weighted_c, axis=0)
pc = pc.reshape(*img_resize[::-1], 3)
pc = (pc-np.min(pc)) / (np.max(pc) - np.min(pc))
axes2[1].imshow(pc)

# -----------------------------------------------------------------------------/
# %%
# 預測訓練集
pred_train = random_forest.predict(input_training)
pred_train = [labels[c_idx] for c_idx in pred_train]

gt_training = list(training_df["class"])

# reports
cls_report = classification_report(y_true=gt_training,
                                   y_pred=pred_train, digits=5)
_, confusion_matrix = confusion_matrix_with_class(prediction=pred_train,
                                                  ground_truth=gt_training)
# display report
print("Classification Report:\n\n", cls_report)
print(f"{confusion_matrix}\n")

# log file
with open(dst_dir.joinpath(f"{notebook_name}.{img_mode}.train.log"), mode="w") as f_writer:
    f_writer.write("Classification Report:\n\n")
    f_writer.write(f"{cls_report}\n\n")
    f_writer.write(f"{confusion_matrix}\n")

# -----------------------------------------------------------------------------/
# %% [markdown]
# ## Test

# -----------------------------------------------------------------------------/
# %%
data = []
for palmskin_dname in tqdm(test_df["palmskin_dname"]):
    img_path: Path = processed_di.palmskin_processed_dir.joinpath(palmskin_dname, rel_path)
    image = cv2.imread(str(img_path))
    if image is not None:
        image = cv2.cvtColor(image, getattr(cv2, f"COLOR_BGR2{img_mode}"))
        image = cv2.resize(image, img_resize, interpolation=cv2.INTER_CUBIC)
        image = image.flatten()
        data.append([image, img_path])

features, img_paths  = zip(*data)

# -----------------------------------------------------------------------------/
# %%
input_test = pca.transform(features)

# 預測測試集
pred_test = random_forest.predict(input_test)
pred_test = [labels[c_idx] for c_idx in pred_test]

gt_test = list(test_df["class"])

# reports
cls_report = classification_report(y_true=gt_test,
                                   y_pred=pred_test, digits=5)
_, confusion_matrix = confusion_matrix_with_class(prediction=pred_test,
                                                  ground_truth=gt_test)
# display report
print("Classification Report:\n\n", cls_report)
print(f"{confusion_matrix}\n")

# log file
with open(dst_dir.joinpath(f"{notebook_name}.{img_mode}.test.log"), mode="w") as f_writer:
    f_writer.write("Classification Report:\n\n")
    f_writer.write(f"{cls_report}\n\n")
    f_writer.write(f"{confusion_matrix}\n")

# -----------------------------------------------------------------------------/
# %%
np.array(pred_test)

# -----------------------------------------------------------------------------/
# %%
np.array(gt_test)