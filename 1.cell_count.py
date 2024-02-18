import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rich import print
from rich.pretty import Pretty
from rich.progress import Progress
from rich.traceback import install

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.shared.clioutput import CLIOutput
from modules.shared.config import dump_config, load_config
from modules.shared.utils import create_new_dir, get_repo_root
from slic_labeling import run_single_slic_analysis

install()
# -----------------------------------------------------------------------------/


def gen_slic_analysis_dict(seg:np.ndarray, merge:int) -> dict[str, Any]:
    """
    """
    tmp_dict: dict = {}
    tmp_key = ""
    if merge > 0: tmp_key = "cell_count_merge"
    else: tmp_key = "cell_count"
    tmp_dict[tmp_key] = len(np.unique(seg))-1
    
    return  tmp_dict
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
    processed_di.parse_config("cell_count.toml")

    # load config
    # `dark` and `merge` are two parameters as color space distance, determined by experiences
    config = load_config("cell_count.toml")
    palmskin_result_name: str = config["data_processed"]["palmskin_result_name"]
    n_segments: int  = config["slic"]["n_segments"]
    dark: int        = config["slic"]["dark"]
    merge: int       = config["slic"]["merge"]
    debug_mode: bool = config["slic"]["debug_mode"]
    print("", Pretty(config, expand_all=True))
    cli_out.divide()

    """ Colloct image file names """
    rel_path, sorted_results_dict = \
        processed_di.get_sorted_results_dict("palmskin", palmskin_result_name)
    result_paths = list(sorted_results_dict.values())
    print(f"Total files: {len(result_paths)}")

    """ Apply SLIC on each image """
    cli_out.divide()
    with Progress() as pbar:
        task = pbar.add_task("[cyan]Processing...", total=len(result_paths))
        
        for result_path in result_paths:
            
            result_name = result_path.stem
            dname_dir = Path(str(result_path).replace(rel_path, ""))
            slic_dir = dname_dir.joinpath(f"SLIC/{result_name}_{{dark_{dark}}}")
            create_new_dir(slic_dir)
            
            print(f"[ {dname_dir.parts[-1]} ]")
            seg_result = run_single_slic_analysis(slic_dir, result_path,
                                                  n_segments, dark, merge,
                                                  debug_mode)
            analysis_dict = gen_slic_analysis_dict(seg_result, merge)
            cli_out.new_line()
            
            # update info to toml file
            toml_file = slic_dir.joinpath(f"{result_name}.ana.toml")
            update_toml_file(toml_file, analysis_dict)
            
            # update pbar
            pbar.advance(task)

    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/