import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from rich import print
from rich.pretty import Pretty
from rich.traceback import install

from modules.data import dname
from modules.data.processeddatainstance import ProcessedDataInstance
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config
from modules.shared.utils import create_new_dir, get_repo_root

install()
# -----------------------------------------------------------------------------/


if __name__ == '__main__':
    
    repo_root = get_repo_root()
    print(f"Repository: '{repo_root}'")
    
    """ Init components """
    cli_out = CLIOutput()
    processed_di = ProcessedDataInstance()
    processed_di.parse_config("ml_analysis.toml")
    # load config
    config = load_config("ml_analysis.toml")
    palmskin_result_name: Path = Path(config["data_processed"]["palmskin_result_name"])
    cluster_desc: str = config["data_processed"]["cluster_desc"]
    dark: int = config["SLIC"]["dark"]
    print("", Pretty(config, expand_all=True))
    cli_out.divide()
    
    # get `slic_dirname`
    slic_dirname = f"{palmskin_result_name.stem}_{{dark_{dark}}}"
    
    # load `clustered file`
    csv_path = processed_di.clustered_files_dict[cluster_desc]
    clustered_df: pd.DataFrame = pd.read_csv(csv_path, encoding='utf_8_sig', index_col=[0])
    clustered_df["fish_id"] = clustered_df["Brightfield"].apply(lambda x: dname.get_dname_sortinfo(x)[0])
    palmskin_dnames = sorted(pd.concat([clustered_df["Palmskin Anterior (SP8)"],
                                        clustered_df["Palmskin Posterior (SP8)"]]), key=dname.get_dname_sortinfo)
    
    # collect informations
    dataset_df = pd.DataFrame()
    for palmskin_dname in palmskin_dnames:
        # prepare
        path = processed_di.palmskin_processed_dir.joinpath(palmskin_dname)
        slic_analysis = path.joinpath("SLIC", slic_dirname,
                                      f"{palmskin_result_name.stem}.ana.toml")
        slic_analysis = load_config(slic_analysis)
        fish_id = dname.get_dname_sortinfo(palmskin_dname)[0]
        
        # >>> Create `temp_dict` <<<
        temp_dict = {}
        # -------------------------------------------------------
        temp_dict["palmskin_dname"] = palmskin_dname
        temp_dict["class"] = clustered_df.loc[fish_id, "class"]
        temp_dict["dataset"] = clustered_df.loc[fish_id, "dataset"]
        # -------------------------------------------------------
        for k, v in slic_analysis.items():
            temp_dict[k] = v
        # -------------------------------------------------------
        
        temp_df = pd.DataFrame(temp_dict, index=[0])
        if dataset_df.empty: dataset_df = temp_df.copy()
        else: dataset_df = pd.concat([dataset_df, temp_df], ignore_index=True)
    
    # save Dataframe as a CSV file
    save_path = repo_root.joinpath("data/generated/ML", processed_di.instance_name,
                                   cluster_desc, slic_dirname, f"ml_dataset.csv")
    create_new_dir(save_path.parent)
    dataset_df.to_csv(save_path, encoding='utf_8_sig', index=False)
    print(f"ML_table: '{save_path}'\n")
    # -------------------------------------------------------------------------/