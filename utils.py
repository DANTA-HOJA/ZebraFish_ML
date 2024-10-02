from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
# -----------------------------------------------------------------------------/


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