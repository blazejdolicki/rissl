import json
import os
import argparse
import numpy as np
import subprocess
import os

# TODO: pass: checkpoint_dir
parser = argparse.ArgumentParser()
parser.add_argument("--subfolder_results_path", type=str, default="results.json")
args = parser.parse_args()

data_params = {
                "bach": {"data_dir": "/mnt/archive/projectdata/data_bach",
                         "val_name": "",
                         "batch_size": 128},
                "breakhis": {"data_dir": "/mnt/archive/projectdata/data_breakhis",
                             "val_name": "val",
                             "batch_size": 128},
                "nct": {"data_dir": "/mnt/archive/projectdata/data_nct",
                        "val_name": "valid",
                        "batch_size": 512},
                "pcam": {"data_dir": "/mnt/archive/projectdata/data_pcam",
                         "val_name": "valid",
                         "batch_size": 512},
               }

log_dirs = [
            # ["/home/b.dolicki/logs/supervised/bach/15271", "bach"],
            # ["/home/b.dolicki/logs/supervised/bach_e2/15324", "bach"],
            # ["/home/b.dolicki/logs/benchmark/finetune/finetune_bach_no_rot_aug/15247", "bach"],
            # ["/home/b.dolicki/logs/benchmark/finetune/finetune_bach/15246", "bach"],
            # ["/home/b.dolicki/logs/benchmark/finetune/finetune_bach_e2/15323", "bach"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_bach_no_rot_aug/15280", "bach"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_bach/15277", "bach"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_bach_e2/15325", "bach"],
            # ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_bach_10perc/15283", "bach"],
            # ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_bach_e2_10perc/15329", "bach"],
            # ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_bach_1perc/15284", "bach"],
            # ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_bach_e2_1perc/15330", "bach"],
            # ["/home/b.dolicki/logs/supervised/breakhis/14455", "breakhis"],
            # ["/home/b.dolicki/logs/supervised/breakhis_e2/14459", "breakhis"],
            # ["/home/b.dolicki/logs/benchmark/finetune/finetune_breakhis_moco_no_rot_aug/14510", "breakhis"],
            # ["/home/b.dolicki/logs/benchmark/finetune/finetune_breakhis_moco/14511", "breakhis"],
            # ["/home/b.dolicki/logs/benchmark/finetune/finetune_breakhis_e2_moco/14683", "breakhis"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_breakhis_moco/14630", "breakhis"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_breakhis_moco_no_rot_aug/14635", "breakhis"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_breakhis_moco_e2/14681", "breakhis"],
            # ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_breakhis_10perc/15075", "breakhis"],
            # ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_breakhis_e2_10perc/15125", "breakhis"],
            # ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_breakhis_1perc/14870", "breakhis"],
            # ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_breakhis_moco_e2_1perc/14986", "breakhis"],
            # ["/home/b.dolicki/logs/supervised/nct/14526", "nct"],
            # ["/home/b.dolicki/logs/supervised/nct_e2/14528", "nct"],
            # ["/home/b.dolicki/logs/benchmark/finetune/finetune_nct_no_rot_aug/14975", "nct"],
            # ["/home/b.dolicki/logs/benchmark/finetune/finetune_nct/14976", "nct"],
            # ["/home/b.dolicki/logs/benchmark/finetune/finetune_nct_e2_moco/14991", "nct"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_nct_moco_no_rot_aug/15009", "nct"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_nct/15007", "nct"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_nct_moco_e2/15010", "nct"],
            # ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_nct_10perc/15210", "nct"],
            # ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_nct_e2_10perc/15173", "nct"],
            # ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_nct_1perc/15073", "nct"],
            # ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_nct_e2_1perc/15208", "nct"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_pcam_moco_no_rot_aug/14762", "pcam"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_pcam_moco/14688", "pcam"],
            ["/home/b.dolicki/logs/benchmark/linear/linear_pcam_moco_e2/14739", "pcam"],
            # ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_pcam_10perc/15171", "pcam"],
            # ["/home/b.dolicki/logs/benchmark/linear/10perc/linear_pcam_moco_e2_10perc/15209", "pcam"],
            # ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_pcam_1perc/14872", "pcam"],
            # ["/home/b.dolicki/logs/benchmark/linear/1perc/linear_pcam_moco_e2_1perc/14985", "pcam"]
            ]
# TODO add finetuning and supervised PCam

job_id = os.environ["SLURM_JOB_ID"]

table_dict = {"method": [], "dataset": [], "acc": [], "auroc": [], "f1": [], "loss": []}
table_split = "test"

for log_dir, dataset in log_dirs:
    data_dir = data_params[dataset]["data_dir"]
    val_name = data_params[dataset]["val_name"]
    batch_size = data_params[dataset]["batch_size"]
    job_name = log_dir.split("/")[-2]

    avg_results = {}
    output_path = os.path.join(log_dir, "avg_seed_all_results.json")

    for seed_folder in os.listdir(log_dir):
        seed_dir = os.path.join(log_dir, seed_folder)
        if os.path.isdir(seed_dir):

            if ("supervised" in seed_dir) or ("finetune" in seed_dir):
                checkpoint_name = "best_model.pt" if dataset != "bach" else "final_model.pt"
                model_weights = os.path.join(seed_dir, "checkpoints", checkpoint_name)
            else:
                checkpoint_name = "converted_best_model.torch" if dataset != "bach" else "converted_final_model.torch"
                model_weights = os.path.join(seed_dir, checkpoint_name)
            eval_dir = os.path.join(seed_dir, "evaluate", job_id)
            subprocess.run(["python", "evaluate.py",
                            "--splits", val_name+",test" if dataset != "bach" else "test",
                            "--dataset", dataset,
                            "--data_dir", data_dir,
                            "--log_dir", eval_dir,
                            "--mlflow_dir", "/home/b.dolicki/mlflow_runs",
                            "--checkpoint_path", model_weights,
                            "--batch_size", str(batch_size),
                            "--num_workers", str(1)
                            ])

            metrics_path = os.path.join(seed_dir, "evaluate", job_id, args.subfolder_results_path)
            with open(metrics_path) as json_file:
                results = json.load(json_file)
            print("results", results)

            for split in results:
                if split not in avg_results:
                    avg_results[split] = {}
                for metric in results[split]:
                    metric_plural = metric + "s" if metric != "loss" else metric + "es"
                    if metric_plural not in avg_results[split]:
                        avg_results[split][metric_plural] = []
                    avg_results[split][metric_plural].append(results[split][metric])

    for split in avg_results:
        for metric in results[split]:
            metric_plural = metric + "s" if metric != "loss" else metric + "es"
            # setting axis=0 is especially important for non-scalar metrics such as confusion matrix
            avg_results[split]["avg_" + metric] = np.mean(avg_results[split][metric_plural], axis=0)
            avg_results[split]["std_" + metric] = np.std(avg_results[split][metric_plural], axis=0)

            if isinstance(results[split][metric], list):
                avg_results[split]["avg_" + metric] = avg_results[split]["avg_" + metric].tolist()
                avg_results[split]["std_" + metric] = avg_results[split]["std_" + metric].tolist()
            elif split == table_split:
                table_dict["Method"] = job_name
                table_dict["Dataset"] = dataset
                table_dict[metric] = avg_results[split]["avg_" + metric]

    with open(output_path, 'w') as file:
        json.dump(avg_results, file, indent=4)

