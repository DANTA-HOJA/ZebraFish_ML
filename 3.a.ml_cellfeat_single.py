# %%
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rich
from rich.pretty import Pretty
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.dl.tester.utils import confusion_matrix_with_class
from modules.shared.config import load_config
from modules.shared.utils import get_repo_root

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
dst_dir = ml_csv.parent

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
feature = config["ML"]["single_feature"]
assert feature in df.columns[3:], \
            f"Feature should be one of followings: {list(df.columns)[3:]}"

# -----------------------------------------------------------------------------/
# %%
input_training = training_df[feature].to_numpy()[:, None]

# 初始化 Random Forest 分類器
rand_seed = int(cluster_desc.split("_")[-1].replace("RND", ""))
random_forest = RandomForestClassifier(n_estimators=100, random_state=rand_seed)

# 訓練模型
idx_gt_training = [label2idx[c_label] for c_label in training_df["class"]]
random_forest.fit(input_training, idx_gt_training)

# get auto tree depths
tree_depths = {}
for i, tree in enumerate(random_forest.estimators_):
    tree_depths[f"Tree {i+1} depth"] = tree.tree_.max_depth

tree_depths["mean depth"] = np.mean(list(tree_depths.values()))
print(f"-> mean of tree depths: {tree_depths['mean depth']}")

tree_depths["median depth"] = np.median(list(tree_depths.values()))
print(f"-> median of tree depths: {tree_depths['median depth']}")

with open(dst_dir.joinpath(f"{notebook_name}.{feature}.tree_depths.log"), mode="w") as f_writer:
    json.dump(tree_depths, f_writer, indent=4)

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
with open(dst_dir.joinpath(f"{notebook_name}.{feature}.train.log"), mode="w") as f_writer:
    f_writer.write("Classification Report:\n\n")
    f_writer.write(f"{cls_report}\n\n")
    f_writer.write(f"{confusion_matrix}\n")

# -----------------------------------------------------------------------------/
# %% [markdown]
# ## Test

# -----------------------------------------------------------------------------/
# %%
input_test = test_df[feature].to_numpy()[:, None]

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
with open(dst_dir.joinpath(f"{notebook_name}.{feature}.test.log"), mode="w") as f_writer:
    f_writer.write("Classification Report:\n\n")
    f_writer.write(f"{cls_report}\n\n")
    f_writer.write(f"{confusion_matrix}\n")

# -----------------------------------------------------------------------------/
# %%
np.array(pred_test)

# -----------------------------------------------------------------------------/
# %%
np.array(gt_test)