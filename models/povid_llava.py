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
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../POVID")))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../POVID/llava")))
from POVID.llava.model.builder import load_pretrained_model
from POVID.llava.utils import disable_torch_init

from PIL import Image

from POVID.llava.conversation import conv_templates,SeparatorStyle
from POVID.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)
from POVID.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,

)


def load_llava_model():
    model_name = get_model_name_from_path(MODEL_ID_POVID)

    tokenizer, model, image_processor, context_len = load_pretrained_model(MODEL_ID_POVID,MODEL_ID_LLAVA, model_name,device='cuda',device_map='cuda')

    return tokenizer, model, image_processor, context_len



def predict_llava_model(tokenizer, model, image_processor,qs,images):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if model.config.mm_use_im_start_end:
        qs = image_token_se + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs


    conv = conv_templates['llava_v1'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    images_tensor = process_images(
        [images],
        image_processor,
        model.config
    )

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor.to(model.device, dtype=torch.float16),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            use_cache=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def MME_eval():
    tokenizer, model, image_processor, context_len = load_llava_model()
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
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_mme_povid.pkl')

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
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_POPE_qa_coco_povid.pkl')

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
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_POPE_qa_aokvqa_povid.pkl')

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
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_POPE_qa_gqa_povid.pkl')

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
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_mmvet_povid.pkl')
    with open(f'{RESULT_PATH}/mmvet_povid.json','w') as f:
        json.dump(res_df.set_index('ant_id',append=False).prediction.to_dict(),f,indent=2)

from benchmarks.MMMU import *
import io
def MMMU_eval():
    tokenizer, model, image_processor, context_len = load_llava_model()
    data_generator = load_dataset_MMMU_qa()
    res = {}
    for d in data_generator:
        if d['image']:
            image_bytes = d["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            prompt=d['final_input_prompt']
            # prompt = prompt.replace("<image 1>", IMAGE_PLACEHOLDER)
            response = predict_llava_model(tokenizer, model, image_processor, prompt, image)
            if d['question_type']=='multiple-choice':
                response=parse_multi_choice_response(response,d['all_choices'],d['index2ans'])
        else:
            if d['question_type'] == 'multiple-choice':
                all_choices = d['all_choices']
                response = random.choice(all_choices)
            else:
                response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'
        res[d['id']] = response

    with open(f'{RESULT_PATH}/eval_res_mmmu_povid.json', 'w') as f:
        json.dump(res, f, indent=4)

def CHAIR_eval():
    tokenizer, model, image_processor, context_len = load_llava_model()

    # Load Karpathy test split
    with open(f"{DPATH_CHAIR}/caption_datasets/dataset_coco.json") as f:
        coco_data = json.load(f)

    test_images = [img for img in coco_data['images'] if img['split'] == 'test']

    results = []
    for img in tqdm(test_images):
        image_id = img['cocoid']
        image_path = os.path.join(DPATH_COCO, img['filename'])

        image = Image.open(image_path).convert("RGB")

        # Standard captioning prompt
        prompt = "Describe this image."
        response = predict_llava_model(tokenizer, model, image_processor, prompt, image)

        results.append({
            "image_id": image_id,
            "caption": response
        })

    # Save the predictions directly for CHAIR evaluation
    with open(f'{RESULT_PATH}/eval_res_chair_povid.json', 'w') as f:
        json.dump(results, f, indent=2)

from benchmarks.Llava_Bench import *
import shortuuid
def LLAVABENCH_eval():
    tokenizer, model, image_processor, context_len = load_llava_model()
    data_generator = load_dataset_llavabench()
    answers_file = os.path.expanduser(f"{RESULT_PATH}/eval_res_llavabench_povid.jsonl")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for d in data_generator:

        images=Image.open(d['image_path']).convert('RGB')
        response=predict_llava_model(tokenizer,model,image_processor,d['q'],images)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": d['question_id'],
                                   "prompt": d['q'],
                                   "text": response,
                                   "answer_id": ans_id,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__=="__main__":
    # Set decoding method here or via argparse/config
    parser = argparse.ArgumentParser(description="POVID Llama evaluation")
    parser.add_argument("--benchmark_name", type=str,default="MME_Bench", help="benchmark name")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
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

    elif args.benchmark_name == 'MMMU':
        MMMU_eval()

    elif args.benchmark_name == 'CHAIR':
        CHAIR_eval()

    elif args.benchmark_name == 'LLAVA_Bench':
        LLAVABENCH_eval()

    elif args.benchmark_name =='all':
        MME_eval()
        POPE_coco_eval()
        POPE_aokvqa_eval()
        POPE_gqa_eval()
        MMVet_eval()





