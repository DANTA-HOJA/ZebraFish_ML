import os
import sys
from pathlib import Path

import numpy as np
from rich import print
from rich.pretty import Pretty
from rich.progress import Progress
from rich.traceback import install

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.shared.clioutput import CLIOutput
from modules.shared.config import dump_config, load_config
from modules.shared.utils import create_new_dir, get_repo_root
from slic_labeling import run_single_slic_process

install()
# -----------------------------------------------------------------------------/


def gen_slic_analysis_dict(seg:np.ndarray, merge:int=0) -> dict[str, any]:
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
    processed_di.set_attrs("cell_count.toml")

    # load config
    # `dark` and `merge` are two parameters as color space distance, determined by experiences
    config = load_config("cell_count.toml")
    palmskin_result_alias = config["data_processed"]["palmskin_result_alias"]
    n_segments = config["slic"]["n_segments"]
    dark       = config["slic"]["dark"]
    merge      = config["slic"]["merge"]
    debug_mode = config["slic"]["debug_mode"]
    print("", Pretty(config, expand_all=True))
    cli_out.divide()

    """ Colloct image file names """
    rel_path, result_paths = \
        processed_di._get_sorted_results("palmskin", palmskin_result_alias)
    print(f"Total files: {len(result_paths)}")

    """ Apply SLIC on each image """
    cli_out.divide()
    with Progress() as pbar:
        task = pbar.add_task("[cyan]Processing...", total=len(result_paths))
        
        for result_path in result_paths:
            
            result_path = str(result_path)
            result_name = os.path.split(result_path)[-1]
            result_name = os.path.splitext(result_name)[0]
            dname_dir = Path(result_path.replace(str(Path(rel_path)), ""))
            slic_dir = dname_dir.joinpath(f"SLIC/{result_name}")
            create_new_dir(slic_dir)
            
            print(f"[ {os.path.split(dname_dir)[-1]} ]")
            seg_result = run_single_slic_process(slic_dir, result_path,
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