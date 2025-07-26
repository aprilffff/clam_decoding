from sklearn.model_selection import ParameterGrid
import os
import numpy as np
# Define parameter space as a dict for sklearn's ParameterGrid
param_grid = {
    "scale": [0.005, 0.01,0.02,0.05],
    "start_layer": [8, 16, 24],
    "reduction": [64, 128, 256]
}
default_clam_param={
    "scale": 0.02,
    "start_layer": 16,
    "reduction": 128
}
model_list=['clam']#,'damo','deco','dola','greedy','opera','povid','vcd','vdd']

# Generate parameter combinations
sklearn_param_combinations = list(ParameterGrid(param_grid))

num_gpu=2

for benchmark in ['POPE_aokvqa','POPE_gqa']:#"MME_Bench",'MMVet','MMMU','LLAVA_Bench','POPE_coco',
    script_lines_cuda0 = [
        "#!/bin/bash",
        "",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "",
    ]
    script_lines_cuda1 = [
        "#!/bin/bash",
        "",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "",
    ]
    model_cnt = 0
    for model in model_list:
        cuda_num=model_cnt%num_gpu
        if model=='clam':
            for param_combo in sklearn_param_combinations:
                cuda_num = model_cnt % num_gpu
                scale = param_combo["scale"]
                start_layer = param_combo["start_layer"]
                reduction = param_combo["reduction"]
                if np.sum([scale==default_clam_param['scale'],start_layer==default_clam_param['start_layer'],reduction==default_clam_param['reduction']])!=2:
                    continue
                model_path=f"/root/autodl-tmp/PycharmProjects/llava-decoding-exp/checkpoints/llava-cla-scale{scale}-reduction{reduction}-startlayer{start_layer}-addnorm-v2"
                if os.path.exists(model_path):
                    globals()[f"script_lines_cuda{cuda_num}"].append(f"CUDA_VISIBLE_DEVICES={cuda_num} python -m models.{model}_llava --benchmark {benchmark} --model_path {model_path}")
                    model_cnt+=1
        else:
            globals()[f"script_lines_cuda{cuda_num}"].append(f"CUDA_VISIBLE_DEVICES={cuda_num} python -m models.{model}_llava --benchmark {benchmark}")
            model_cnt+=1


    script_path_cuda0 = f"/root/autodl-tmp/PycharmProjects/llava-decoding-exp/scripts/run_eval_{benchmark}_cuda0.sh"
    with open(script_path_cuda0, "w") as f:
        f.write("\n".join(script_lines_cuda0))

    script_path_cuda1 = f"/root/autodl-tmp/PycharmProjects/llava-decoding-exp/scripts/run_eval_{benchmark}_cuda1.sh"
    with open(script_path_cuda1, "w") as f:
        f.write("\n".join(script_lines_cuda1))