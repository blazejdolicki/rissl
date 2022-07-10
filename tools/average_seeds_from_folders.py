import json
from pathlib import Path
import numpy as np

folders = [["/home/b.dolicki/logs/benchmark/finetune/finetune_pcam_e2_moco/14364/",
            "/home/b.dolicki/logs/benchmark/finetune/finetune_pcam_e2_moco/14416/",
            "/home/b.dolicki/logs/benchmark/finetune/finetune_pcam_e2_moco/14421/"],
           ["/home/b.dolicki/logs/benchmark/finetune/finetune_pcam_moco/14244/",
            "/home/b.dolicki/logs/benchmark/finetune/finetune_pcam_moco/14301/",
            "/home/b.dolicki/logs/benchmark/finetune/finetune_pcam_moco/14414/"],
           ["/home/b.dolicki/logs/benchmark/finetune/finetune_pcam_moco_no_rot_aug/14363/",
            "/home/b.dolicki/logs/benchmark/finetune/finetune_pcam_moco_no_rot_aug/14415",
            "/home/b.dolicki/logs/benchmark/finetune/finetune_pcam_moco_no_rot_aug/14439"]]

for i in range(len(folders)):
    avg_results = {}
    output_name = []
    for folder in folders[i]:
        folder_path = Path(folder)
        folder_name = folder_path.parts[-1]
        output_name.append(folder_name)


        eval_path = folder_path / "evaluate"
        first_eval_child = next(eval_path.iterdir())
        assert next(eval_path.iterdir()) == first_eval_child, "There should be only one child in directory"
        acc_path = first_eval_child / "results.json"

        with open(acc_path) as json_file:
            results = json.load(json_file)

        for split in results:
            if split not in avg_results:
                avg_results[split] = {}
                avg_results[split]["accs"] = []
            avg_results[split]["accs"].append(results[split]["acc"])



        mre_path = folder_path / "mre"
        if mre_path.exists():
            first_mre_child = next(mre_path.iterdir())
            assert next(first_mre_child.iterdir()) is None, "There should be only one child in directory"
            mre_path = first_mre_child / "results.json"

            with open(mre_path) as json_file:
                results = json.load(json_file)

            for split in results:
                avg_results[split]["mre"].append(results[split]["mre"])


    for split in avg_results:
        avg_results[split]["avg_acc"] = np.mean(avg_results[split]["accs"])
        avg_results[split]["std_acc"] = np.std(avg_results[split]["accs"])

    output_name = "avg_"+"_".join(output_name) + ".json"

    output_path = Path(folders[i][0]).parent / output_name
    with open(output_path, 'w') as file:
        json.dump(avg_results, file, indent=4)
