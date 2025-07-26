import argparse
import pandas as pd
from PIL import Image
import torch
from benchmarks.MME_Benchmark import load_dataset_mme
from benchmarks.POPE import *
from benchmarks.MM_Vet import *
import numpy as np
import sys, os,re
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../")))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../OPERA")))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../OPERA/minigpt4")))
from minigpt4.common.config import Config
from minigpt4.common.registry import registry



from OPERA.pope_eval import parse_args,MODEL_EVAL_CONFIG_PATH,setup_seeds,load_preprocess,INSTRUCTION_TEMPLATE
def load_llava_model():


    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    args.cfg_path = os.path.abspath(os.path.join(__file__, "../OPERA/eval_configs/llava-1.5_eval.yaml"))
    cfg = Config(args)

    setup_seeds(cfg)
    device = "cuda"

    # ========================================
    #             Model Initialization
    # ========================================
    print('Initializing Model')

    model_config = cfg.model_cfg
    # model_config.device_8bit = 0
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()
    vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)
    # vis_processors.do_normalize = False
    # print(vis_processors["eval"].transform)
    # print("Done!")


    return vis_processors, txt_processors, model



def predict_llava_model(vis_processors, txt_processors, model,qs,images):
    template = INSTRUCTION_TEMPLATE['llava-1.5']
    qu = template.replace("<question>", qs)
    images=vis_processors['eval'](images).unsqueeze(0)
    # model=model.to('cuda')#FIXME some operands are not implemented on mps, must use cuda
    images = images.to(model.device, dtype=torch.float16)

    with torch.inference_mode():
        with torch.no_grad():
            outputs = model.generate(
                {"image": images, "prompt": qu},
                use_nucleus_sampling=args.sample,
                num_beams=args.beam,
                max_new_tokens=args.max_new_tokens,
                output_attentions=True,
                opera_decoding=True,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
            )

    return outputs[0]

def MME_eval():
    tokenizer, model, image_processor= load_llava_model()
    data_generator = load_dataset_mme()
    res = []
    for d in data_generator:

        images=Image.open(d['image_path']).convert('RGB')
        response=predict_llava_model(tokenizer,model,image_processor,d['q'],images)
        res.append({
            'category': d['category'],
            'image_name': d['image_name'],
            "question": d['q'],
            'prediction':response,
            'label':d['a']
        })

    res_df = pd.DataFrame(res)
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_mme_opera.pkl')

def POPE_coco_eval():
    tokenizer, model, image_processor, context_len = load_llava_model()
    data_generator = load_dataset_POPE_qa_coco()
    res = []
    for d in data_generator:

        images=Image.open(d['image_path']).convert('RGB')
        response=predict_llava_model(tokenizer,model,image_processor,d['q'],images)

        res.append({
            'category': d['annotation_type'],
            'image_name': d['image_name'],
            "question": d['q'],
            'prediction': response,
            'label': d['a']
        })

    res_df = pd.DataFrame(res)
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_POPE_qa_coco_opera.pkl')

def POPE_aokvqa_eval():
    tokenizer, model, image_processor, context_len = load_llava_model()
    data_generator = load_dataset_POPE_qa_aokvqa()
    res = []
    for d in data_generator:
        images=Image.open(d['image_path']).convert('RGB')
        response=predict_llava_model(tokenizer,model,image_processor,d['q'],images)
        response=response.strip()
        res.append({
            'category': d['annotation_type'],
            'image_name': d['image_name'],
            # 'halluciated': is_hallucinated(response, d['a']),
            "question": d['q'],
            'prediction': response,
            'label': d['a']
        })

    res_df = pd.DataFrame(res)
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_POPE_qa_aokvqa_opera.pkl')

def POPE_gqa_eval():
    tokenizer, model, image_processor, context_len = load_llava_model()
    data_generator = load_dataset_POPE_qa_gqa()
    res = []
    for d in data_generator:
        images=Image.open(d['image_path']).convert('RGB')
        response=predict_llava_model(tokenizer,model,image_processor,d['q'],images)
        response=response.strip()
        res.append({
            'category': d['annotation_type'],
            'image_name': d['image_name'],
            "question": d['q'],
            'prediction': response,
            'label': d['a']
        })

    res_df = pd.DataFrame(res)
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_POPE_qa_gqa_opera.pkl')

def MMVet_eval():
    tokenizer, model, image_processor, context_len = load_llava_model()
    data_generator = load_dataset_mmvet()
    res = []
    for d in data_generator:
        images=Image.open(d['image_path']).convert('RGB')
        response=predict_llava_model(tokenizer,model,image_processor,d['q'],images)
        response=response.strip()
        res.append({
            'ant_id':d['ant_id'],
            'capability': ','.join(d['capability']),
            'image_name': d['image_name'],
            "question":d['q'],
            'prediction': response,
            'label': d['a']
        })

    res_df = pd.DataFrame(res)
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_mmvet_opera.pkl')
    with open(f'{RESULT_PATH}/mmvet_opera.json','w') as f:
        json.dump(res_df.set_index('ant_id',append=False).prediction.to_dict(),f,indent=2)



if __name__=="__main__":
    # Set decoding method here or via argparse/config
    parser = argparse.ArgumentParser(description="OPERA Llama evaluation")

    parser.add_argument("--benchmark_name", type=str,default="MME_Bench", help="benchmark name")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--model", type=str, help="model",default='llava-v1.5')
    # parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--beam", type=int,default=5)
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--scale_factor", type=float, default=50)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--num_attn_candidates", type=int, default=5)
    parser.add_argument("--penalty_weights", type=float, default=1.0)
    args = parser.parse_args()

    print(args)

    if args.benchmark_name == 'MME_Bench':
        MME_eval()

    elif args.benchmark_name == 'POPE_coco':
        POPE_coco_eval()

    elif args.benchmark_name == 'POPE_aokvqa':
        POPE_aokvqa_eval()

    elif args.benchmark_name == 'POPE_gqa':
        POPE_gqa_eval()

    elif args.benchmark_name == 'MMVet':
        MMVet_eval()

    elif args.benchmark_name =='all':
        MME_eval()
        POPE_coco_eval()
        POPE_aokvqa_eval()
        POPE_gqa_eval()
        MMVet_eval()





