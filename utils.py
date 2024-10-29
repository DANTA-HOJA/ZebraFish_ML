from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
# -----------------------------------------------------------------------------/


def get_slic_param_name(config: dict) -> str:
    """
    """
    n_segments: int  = config["SLIC"]["n_segments"]
    dark: int        = config["SLIC"]["dark"]
    merge: int       = config["SLIC"]["merge"]
    
    return f"S{n_segments}_D{dark}_M{merge}"
    # -------------------------------------------------------------------------/


def get_class_weight_dict(dataset_df:pd.DataFrame,
                          labels: list, label2idx: dict[str, int]):
    """
    """
    counter = Counter(dataset_df["class"])
    
    # rearrange dict
    class_counts_dict: dict[str, int] = {}
    for cls in labels:
        class_counts_dict[cls] = counter[cls]
    
    class_weights_dict: dict[int, int] = {}
    total_samples = sum(class_counts_dict.values())
    
    for key, value in class_counts_dict.items(): # value = number of samples of the class
        class_weights_dict[label2idx[key]] = (1 - (value/total_samples))
    
    return class_weights_dict
    # -------------------------------------------------------------------------/


def save_confusion_matrix_display(y_true: list[str],
                                  y_pred: list[str],
                                  save_path: Path,
                                  feature_desc: str,
                                  dataset_desc: str):
    """
    save 2 file:
    1. `save_path`/`feature_desc`.cm.png
    2. `save_path`/`feature_desc`.cm.normgt.png
    """
    ConfusionMatrixDisplay.from_predictions(y_true=y_true,
                                            y_pred=y_pred)
    plt.tight_layout()
    plt.ylabel("Ground truth")
    plt.xlabel("Prediction")
    plt.savefig(save_path.joinpath(f"{feature_desc}.{dataset_desc}.cm.png"))
    plt.savefig(save_path.joinpath(f"{feature_desc}.{dataset_desc}.cm.svg"))
    
    # normalized by row (summation of ground truth samples)
    ConfusionMatrixDisplay.from_predictions(y_true=y_true,
                                            y_pred=y_pred,
                                            normalize="true",
                                            im_kw={"vmin":0.0, "vmax":1.0})
    plt.tight_layout()
    plt.ylabel("Ground truth")
    plt.xlabel("Prediction")
    plt.savefig(save_path.joinpath(f"{feature_desc}.{dataset_desc}.cm.normgt.png"))
    plt.savefig(save_path.joinpath(f"{feature_desc}.{dataset_desc}.cm.normgt.svg"))
    # -------------------------------------------------------------------------/