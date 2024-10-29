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
from modules.shared.utils import create_new_dir
from utils import get_slic_param_name, save_confusion_matrix_display

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
rich.print("", Pretty(config, expand_all=True))

# -----------------------------------------------------------------------------/
# %%
processed_di = ProcessedDataInstance()
processed_di.parse_config("ml_analysis.toml")

# src
slic_param_name = get_slic_param_name(config)
slic_dirname = f"{palmskin_result_name.stem}.{slic_param_name}"
ml_csv = Path(__file__).parent.joinpath("data/generated/ML",
                                        processed_di.instance_name,
                                        cluster_desc, slic_dirname,
                                        "ml_dataset.csv")

# dst dir
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
input_training = training_df.iloc[:, 3:].to_numpy()
dst_dir = dst_dir.joinpath(notebook_name)
create_new_dir(dst_dir)

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
rich.print(f"-> mean of tree depths: {tree_depths['mean depth']}")

tree_depths["median depth"] = np.median(list(tree_depths.values()))
rich.print(f"-> median of tree depths: {tree_depths['median depth']}")

with open(dst_dir.joinpath(f"{notebook_name}.tree_depths.log"), mode="w") as f_writer:
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
rich.print(f"{confusion_matrix}\n")

# log file
with open(dst_dir.joinpath(f"{notebook_name}.train.log"), mode="w") as f_writer:
    f_writer.write("Classification Report:\n\n")
    f_writer.write(f"{cls_report}\n\n")
    f_writer.write(f"{confusion_matrix}\n")

# -----------------------------------------------------------------------------/
# %% [markdown]
# ## Test

# -----------------------------------------------------------------------------/
# %%
input_test = test_df.iloc[:, 3:].to_numpy()

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
rich.print(f"{confusion_matrix}\n")

# log file
with open(dst_dir.joinpath(f"{notebook_name}.test.log"), mode="w") as f_writer:
    f_writer.write("Classification Report:\n\n")
    f_writer.write(f"{cls_report}\n\n")
    f_writer.write(f"{confusion_matrix}\n")

# Confusion Matrix (image ver.)
save_confusion_matrix_display(y_true=gt_test,
                              y_pred=pred_test,
                              save_path=dst_dir,
                              feature_desc=notebook_name,
                              dataset_desc="test")

print(f"Results Save Dir: '{dst_dir}'")

# -----------------------------------------------------------------------------/
# %%
np.array(pred_test)

# -----------------------------------------------------------------------------/
# %%
np.array(gt_test)