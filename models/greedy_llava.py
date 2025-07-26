import argparse
import io

import pandas as pd
# from transformers import AutoProcessor, AutoModelForVision2Seq,LlavaForConditionalGeneration,AutoTokenizer
from PIL import Image
import torch
from benchmarks.MME_Benchmark import load_dataset_mme
from benchmarks.POPE import *
from benchmarks.MM_Vet import *
from benchmarks.MMMU import *
import numpy as np
import sys, os,re
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../../LLaVA")))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../")))

from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)


def load_llava_model():
    model_name = get_model_name_from_path(MODEL_ID_LLAVA)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_ID_LLAVA, None, model_name,device='cuda',device_map='cuda'
    )

    return tokenizer, model, image_processor, context_len



def predict_llava_model(tokenizer, model, image_processor,qs,images):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs


    conv = conv_templates[args.template_version].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    images_tensor = process_images(
        [images],
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    if args.analyze:
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=images_tensor,
                do_sample=args.do_sample,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=True,
                output_hidden_states=True
            )
        output_text=tokenizer.batch_decode(output_ids['sequences'], skip_special_tokens=True)[0].strip()
        res_dict={
                    "prompt": prompt,
                    "input_ids": input_ids.cpu(),
                    # "image_tensor": images_tensor.cpu(),  # optional
                    "generated_text": output_text,
                    # "ground_truth": gt_answer,
                    "hidden_states": torch.concat([t[1][0].cpu() for t in output_ids['hidden_states']],dim=0),
                    "attentions":  torch.cat([t.mean(0).mean(0).mean(0,keepdim=True).cpu() for t in output_ids['attentions'][0]]),
                    "image_start_pos":np.argwhere(input_ids.cpu()==IMAGE_TOKEN_INDEX)[1][0],
                    "image_end_pos":np.argwhere(input_ids.cpu()==IMAGE_TOKEN_INDEX)[1][0]+576,
                    "qs_start_pos":-(input_ids.shape[1]-np.argwhere(input_ids.cpu()==IMAGE_TOKEN_INDEX)[1][0]-1),
                    }
        return res_dict
    else:
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=images_tensor,
                do_sample=args.do_sample,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
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
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_mme_vanilla.pkl')

def POPE_coco_eval():
    tokenizer, model, image_processor, context_len = load_llava_model()
    data_generator = load_dataset_POPE_qa_coco()
    res = []

    if args.analyze:
        vanilla_selection = pd.read_pickle(RESULT_PATH + '/eval_res_POPE_qa_coco_vanilla.pkl')
        clam_selection = pd.read_pickle(RESULT_PATH + f'/eval_res_POPE_qa_coco_cla-scale0.02-reduction128-startlayer16-addnorm-v2.pkl')
    for d in data_generator:
        if args.analyze:
            vanilla_sample=vanilla_selection.loc[(vanilla_selection.category==d['annotation_type'])&(vanilla_selection.image_name==d['image_name'])&(vanilla_selection.question==d['q'])].iloc[0]
            clam_sample=clam_selection.loc[(clam_selection.category==d['annotation_type'])&(clam_selection.image_name==d['image_name'])&(clam_selection.question==d['q'])].iloc[0]
            vanilla_wrong=vanilla_sample.prediction.split(',')[0].lower() != vanilla_sample.label
            clam_correct=clam_sample.prediction.split(',')[0].lower() == clam_sample.label
            if not (vanilla_wrong and clam_correct):
                continue
            images = Image.open(d['image_path']).convert('RGB')
            response = predict_llava_model(tokenizer, model, image_processor, d['q'], images)
            res_dict=response
            res_dict['ground_truth']=d['a']
            res_dict['image_name']=d['image_name']
            res_dict['annotation_type'] = d['annotation_type']
        else:

            images = Image.open(d['image_path']).convert('RGB')
            response = predict_llava_model(tokenizer, model, image_processor, d['q'], images)
            res_dict={
            'category': d['annotation_type'],
            'image_name': d['image_name'],
            "question": d['q'],
            'prediction': response,
            'label': d['a']
        }
        res.append(res_dict)
    if args.analyze:
        analyze_dict={f'sample_{i}':k for i,k in enumerate(res)}
        torch.save(analyze_dict, RESULT_PATH+"/vanilla_output.pt")
    else:
        res_df = pd.DataFrame(res)
        res_df.to_pickle(f'{RESULT_PATH}/eval_res_POPE_qa_coco_vanilla.pkl')

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
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_POPE_qa_aokvqa_vanilla.pkl')

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
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_POPE_qa_gqa_vanilla.pkl')

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
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_mmvet_vanilla.pkl')
    with open(f'{RESULT_PATH}/mmvet_vanilla.json','w') as f:
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

    with open(f'{RESULT_PATH}/eval_res_mmmu_vanilla.json', 'w') as f:
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
    with open(f'{RESULT_PATH}/eval_res_chair_vanilla.json', 'w') as f:
        json.dump(results, f, indent=2)


from benchmarks.Llava_Bench import *
import shortuuid
def LLAVABENCH_eval():
    tokenizer, model, image_processor, context_len = load_llava_model()
    data_generator = load_dataset_llavabench()
    answers_file = os.path.expanduser(f"{RESULT_PATH}/eval_res_llavabench_vanilla.jsonl")
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
    parser = argparse.ArgumentParser(description="vanilla Llama evaluation")
    parser.add_argument("--benchmark_name", type=str,default="POPE_coco", help="benchmark name")
    parser.add_argument("--template_version", type=str, default='llava_v1')
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=int, default=0)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--analyze", type=bool, default=True)
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
        MMMU_eval()





