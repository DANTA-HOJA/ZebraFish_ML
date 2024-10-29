import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich import print
from rich.pretty import Pretty
from rich.progress import Progress
from rich.traceback import install

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.dl.dataset.augmentation import crop_base_size
from modules.shared.clioutput import CLIOutput
from modules.shared.config import dump_config, load_config
from modules.shared.utils import create_new_dir, get_repo_root
from slic_labeling import run_single_slic_analysis
from utils import get_slic_param_name

install()
# -----------------------------------------------------------------------------/


def count_area(seg:np.ndarray) -> tuple[int, str]:
    """ unit: pixel
    """
    foreground = seg > 0
    return int(np.sum(foreground)), "area"
    # -------------------------------------------------------------------------/


def count_element(seg:np.ndarray, key:str) -> tuple[int, str]:
    """
    """
    return len(np.unique(seg[seg > 0])), f"{key}_count"
    # -------------------------------------------------------------------------/


def count_average_size(analysis_dict:dict[str, Any], key:str) -> tuple[float, str]:
    """ unit: pixel
    """
    analysis_dict = deepcopy(analysis_dict)
    area = analysis_dict["area"]
    element_cnt = analysis_dict[f"{key}_count"]
    
    if element_cnt == 0:
        average_size = 0
    else:
        average_size = float(round(area/element_cnt, 5))
    
    return  average_size, f"{key}_avg_size"
    # -------------------------------------------------------------------------/


def get_max_patch_size(seg:np.ndarray) -> tuple[int, str]:
    """ (deprecated) unit: pixel
    
        Note: this function is replaced by `get_patch_size()`
    """
    max_size = 0
    
    labels = np.unique(seg)
    for label in labels:
        if label != 0:
            bw = (seg == label)
            size = np.sum(bw)
            if size > max_size:
                max_size = size
    
    return int(max_size), "max_patch_size"
    # -------------------------------------------------------------------------/


def get_patch_sizes(seg:np.ndarray) -> tuple[list[int], str]:
    """ unit: pixel
    """
    tmp = seg.flatten().tolist()
    tmp = Counter(tmp)
    
    try: # remove background
        tmp.pop(0)
    except KeyError:
        print("Warning: background label missing")
    
    tmp = dict(sorted(tmp.items(), reverse=True, key=lambda x: x[1]))
    
    return list(tmp.values()), f"patch_sizes"
    # -------------------------------------------------------------------------/


def update_slic_analysis_dict(analysis_dict:dict[str, Any], value:Any, key:str) -> dict[str, Any]:
    """
    """
    analysis_dict = deepcopy(analysis_dict)
    analysis_dict[key] = value
    
    return  analysis_dict
    # -------------------------------------------------------------------------/


def update_toml_file(toml_path:Path, analysis_dict:dict):
    """
    """
    if toml_path.exists():
        save_dict = load_config(toml_path)
        for k, v in analysis_dict.items():
            save_dict[k] = v
    else:
        save_dict = analysis_dict
    
    dump_config(toml_path, save_dict)
    # -------------------------------------------------------------------------/


if __name__ == '__main__':

    print(f"Repository: '{get_repo_root()}'")

    """ Init components """
    cli_out = CLIOutput()
    cli_out.divide()
    processed_di = ProcessedDataInstance()
    processed_di.parse_config("ml_analysis.toml")
    w512h1024_cropper = crop_base_size(512, 1024)

    # load config
    # `dark` and `merge` are two parameters as color space distance, determined by experiences
    config = load_config("ml_analysis.toml")
    palmskin_result_name: Path = Path(config["data_processed"]["palmskin_result_name"])
    n_segments: int  = config["SLIC"]["n_segments"]
    dark: int        = config["SLIC"]["dark"]
    merge: int       = config["SLIC"]["merge"]
    debug_mode: bool = config["SLIC"]["debug_mode"]
    print("", Pretty(config, expand_all=True))
    cli_out.divide()

    # get `slic_dirname`
    slic_param_name = get_slic_param_name(config)
    slic_dirname = f"{palmskin_result_name.stem}.{slic_param_name}"
    
    """ Colloct image file names """
    rel_path, sorted_results_dict = \
        processed_di.get_sorted_results_dict("palmskin", str(palmskin_result_name))
    result_paths = list(sorted_results_dict.values())
    print(f"Total files: {len(result_paths)}")

    """ Apply SLIC on each image """
    cli_out.divide()
    with Progress() as pbar:
        task = pbar.add_task("[cyan]Processing...", total=len(result_paths))
        
        for result_path in result_paths:
            
            dname_dir = Path(str(result_path).replace(rel_path, ""))
            print(f"[ {dname_dir.parts[-1]} ]")
            
            # create `slic_dir`
            slic_dir = dname_dir.joinpath(f"SLIC/{slic_dirname}")
            create_new_dir(slic_dir)
            
            # get image, size: W512_H1024 (FixedROI)
            target_path = slic_dir.parent.joinpath(f"{result_path.stem}.W512_H1024.tif")
            if not target_path.exists():
                tmp_img = w512h1024_cropper(image=cv2.imread(str(result_path)))
                cv2.imwrite(str(target_path), tmp_img)
            
            # slic
            cell_seg, patch_seg = run_single_slic_analysis(slic_dir, target_path,
                                                            n_segments, dark, merge,
                                                            debug_mode)
            
            # update
            analysis_dict = {}
            analysis_dict = update_slic_analysis_dict(analysis_dict, *count_area(cell_seg))
            analysis_dict = update_slic_analysis_dict(analysis_dict, *count_element(cell_seg, "cell"))
            analysis_dict = update_slic_analysis_dict(analysis_dict, *count_element(patch_seg, "patch"))
            analysis_dict = update_slic_analysis_dict(analysis_dict, *count_average_size(analysis_dict, "cell"))
            analysis_dict = update_slic_analysis_dict(analysis_dict, *count_average_size(analysis_dict, "patch"))
            analysis_dict = update_slic_analysis_dict(analysis_dict, *get_patch_sizes(patch_seg))
            cli_out.new_line()
            
            # update info to toml file
            toml_file = slic_dir.joinpath(f"{slic_dirname}.ana.toml")
            update_toml_file(toml_file, analysis_dict)
            
            # update pbar
            pbar.advance(task)

    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/