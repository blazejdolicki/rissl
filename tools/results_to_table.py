import pandas as pd
import os
import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, choices=["valid", "test"], default="test")
args = parser.parse_args()

data_params = {
                "bach": {"data_dir": "/mnt/archive/projectdata/data_bach",
                         "val_name": ""},
                "breakhis": {"data_dir": "/mnt/archive/projectdata/data_breakhis",
                             "val_name": "val"},
                "nct": {"data_dir": "/mnt/archive/projectdata/data_nct",
                        "val_name": "valid"},
                "pcam": {"data_dir": "/mnt/archive/projectdata/data_pcam",
                         "val_name": "valid"},
               }

log_dirs = [
            ["/home/b.dolicki/logs/supervised/bach/15271", "bach"],
            ["/home/b.dolicki/logs/supervised/bach_e2/15324", "bach"],
            ["/home/b.dolicki/logs/benchmark/finetune/finetune_bach_no_rot_aug/15247", "bach"],
            ["/home/b.dolicki/logs/benchmark/finetune/finetune_bach/15246", "bach"],
            ["/home/b.dolicki/logs/benchmark/finetune/finetune_bach_e2/15323", "bach"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_bach_no_rot_aug/15280", "bach"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_bach/15277", "bach"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_bach_e2/15325", "bach"],
            ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_bach_10perc/15283", "bach"],
            ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_bach_e2_10perc/15329", "bach"],
            ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_bach_1perc/15284", "bach"],
            ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_bach_e2_1perc/15330", "bach"],
            ["/home/b.dolicki/logs/supervised/breakhis/14455", "breakhis"],
            ["/home/b.dolicki/logs/supervised/breakhis_e2/14459", "breakhis"],
            ["/home/b.dolicki/logs/benchmark/finetune/finetune_breakhis_moco_no_rot_aug/14510", "breakhis"],
            ["/home/b.dolicki/logs/benchmark/finetune/finetune_breakhis_moco/14511", "breakhis"],
            ["/home/b.dolicki/logs/benchmark/finetune/finetune_breakhis_e2_moco/14683", "breakhis"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_breakhis_moco/14630", "breakhis"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_breakhis_moco_no_rot_aug/14635", "breakhis"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_breakhis_moco_e2/14681", "breakhis"],
            ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_breakhis_10perc/15075", "breakhis"],
            ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_breakhis_e2_10perc/15125", "breakhis"],
            ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_breakhis_1perc/14870", "breakhis"],
            ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_breakhis_moco_e2_1perc/14986", "breakhis"],
            ["/home/b.dolicki/logs/supervised/nct/14526", "nct"],
            ["/home/b.dolicki/logs/supervised/nct_e2/14528", "nct"],
            ["/home/b.dolicki/logs/benchmark/finetune/finetune_nct_no_rot_aug/14975", "nct"],
            ["/home/b.dolicki/logs/benchmark/finetune/finetune_nct/14976", "nct"],
            ["/home/b.dolicki/logs/benchmark/finetune/finetune_nct_e2_moco/14991", "nct"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_nct_moco_no_rot_aug/15009", "nct"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_nct/15007", "nct"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_nct_moco_e2/15010", "nct"],
            ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_nct_10perc/15210", "nct"],
            ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_nct_e2_10perc/15173", "nct"],
            ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_nct_1perc/15073", "nct"],
            ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_nct_e2_1perc/15208", "nct"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_pcam_moco_no_rot_aug/14762", "pcam"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_pcam_moco/14688", "pcam"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_pcam_moco_e2/14739", "pcam"],
            ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_pcam_10perc/15171", "pcam"],
            ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_pcam_moco_e2_10perc/15209", "pcam"],
            ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_pcam_1perc/14872", "pcam"],
            ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_pcam_moco_e2_1perc/14985", "pcam"]
            ]

job_id = os.environ["SLURM_JOB_ID"]

table_dict = defaultdict(list)
for log_dir, dataset in log_dirs:
    if args.split == "valid":
        if dataset == "bach":
            continue
        else:
            split = data_params[dataset]["val_name"]
    else:
        split = args.split

    job_name = log_dir.split("/")[-2]
    with open(os.path.join(log_dir, "avg_seed_all_results.json")) as json_file:
        avg_results = json.load(json_file)

    table_dict["method"].append(job_name)
    table_dict["dataset"].append(dataset)
    for metric in ["acc", "auroc", "f1", "loss"]:
        table_dict["avg_" + metric].append(avg_results[split]["avg_" + metric])
        table_dict["std_" + metric].append(avg_results[split]["std_" + metric])
        


table = pd.DataFrame(table_dict)
table_dir = "/home/b.dolicki/logs/final_table/"
table_path = os.path.join(table_dir, job_id+"_"+args.split+".csv")
os.makedirs(table_dir, exist_ok=True)
table.to_csv(table_path, index=False)
print(table)