import os
openai_token=""
root_folder_path=os.path.abspath(os.path.join(__file__, "../../../"))

MODEL_ID_LLAVA=f"{root_folder_path}/llm-param-checkpoints/llava-v1.5-7b"
MODEL_ID_LLAMA=f"{root_folder_path}/llm-param-checkpoints/Meta-Llama-3.1-8B"
MODEL_ID_POVID=f"{root_folder_path}/llm-param-checkpoints/llava_POVID_stage_two_lora"#/checkpoint-14000"
MODEL_CLIP=f"{root_folder_path}/llm-param-checkpoints/clip-vit-large-patch14-336"
MODEL_CLAM=f"{root_folder_path}/llava-decoding-exp/checkpoints/llava-cla-scale0.01-reduction256-startlayer16"
DPATH_MME=f"{root_folder_path}/benchmark/MME_Benchmark"
DPATH_COCO=f"{root_folder_path}/benchmark/val2014"
DPATH_POPE=f"{root_folder_path}/benchmark/POPE"
DPATH_GQA=f"{root_folder_path}/benchmark/raw/images"
DPATH_MMVET=f"{root_folder_path}/benchmark/mm-vet"
DPATH_MMMU=f"{root_folder_path}/benchmark/MMMU"
DPATH_CHAIR=f"{root_folder_path}/benchmark/CHAIR"
DPATH_LLAVABENCH=f"{root_folder_path}/benchmark/llava-bench"
DPATH_LLAVABENCH_RULE=f"{root_folder_path}/llava-decoding-exp/eval_tool/rule.json"


RESULT_PATH=f"{root_folder_path}/llava-decoding-exp/result_autodl"


