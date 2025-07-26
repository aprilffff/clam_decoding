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
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../VDD")))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../VDD/experiments/llava")))
from VDD.experiments.llava.model.builder import load_pretrained_model
from VDD.experiments.llava.utils import disable_torch_init
from VDD.experiments.utils.metrics import ECELoss, eval_accuracy, calibrate_label_dict, get_prob_from_logits

from PIL import Image
import math

# import kornia
from transformers import set_seed
from VDD.vcd_utils.vcd_add_noise import add_diffusion_noise
from VDD.vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

from VDD.experiments.llava.conversation import conv_templates,SeparatorStyle
from VDD.experiments.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)
from VDD.experiments.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,

)
LABEL_DICT = {0: ['yes'], 1: ['no']}
LABEL_TO_INT = {'yes': 0, 'no': 1}

def calibrate_label_space(questions, model, tokenizer, images=None, label_dict=LABEL_DICT, top_k=100,
                          content_free_inputs=('N/A',), use_tqdm=True):
    all_p_y = []
    questions = tqdm(questions) if use_tqdm else questions
    for line in questions:
        qs = line["text"]
        if images is not None:
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        if images == 'unk':
            input_ids[input_ids == IMAGE_TOKEN_INDEX] = tokenizer.unk_token_id
            images = None
        with torch.inference_mode():
            images = images.to(dtype=torch.float16, device='mps', non_blocking=True) if images is not None else None
            model_outputs = model.generate(
                input_ids,
                images=images,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,

                cd_beta=args.cd_beta,
                cd_alpha=args.cd_alpha,
                output_scores=True,
                return_dict_in_generate=True)
            output_ids = model_outputs['sequences']
            scores = model_outputs['scores'][0]
        probs_w_token = calibrate_label_dict(scores, tokenizer, apply_softmax=True)
        all_p_y.append(get_prob_from_logits(probs_w_token))
    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y)  # normalize
    if not use_tqdm:
        return p_y, probs_w_token, None, input_ids
    return p_y, None


def load_llava_model():
    model_name = get_model_name_from_path(MODEL_ID_LLAVA)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_ID_LLAVA, None, model_name,device='cuda',device_map='cuda'
    )

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
    if args.use_cd:
        image_tensor_cd = add_diffusion_noise(images_tensor, args.noise_step)
    else:
        image_tensor_cd = None

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        model_outputs = model.generate(
            input_ids,
            images=images_tensor.to(model.device, dtype=torch.float16),
            images_cd=(image_tensor_cd.unsqueeze(0).to(model.device, dtype=torch.float16) if image_tensor_cd is not None else None),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,

            cd_beta=args.cd_beta,
            cd_alpha=args.cd_alpha,
            use_dd=args.use_dd,
            use_dd_unk=args.use_dd_unk,
            output_scores=True,
            return_dict_in_generate=True
        )
        output_ids = model_outputs['sequences']
        # scores = model_outputs['scores'][0]


        # tokens_naive = calibrate_label_dict(scores, tokenizer)
        #
        # image_noise = add_diffusion_noise(images_tensor, 999)
        # image_zero = torch.zeros_like(images_tensor)
        # image_one = torch.ones_like(images_tensor)
        #
        # p_c_none, tokens_none, attention_none, input_ids_none = calibrate_label_space([line], model, tokenizer, use_tqdm=False)
        #
        # p_c_unk, tokens_unk, attention_unk, _ = calibrate_label_space([line], model, tokenizer, images='unk',use_tqdm=False)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
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
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_mme_vdd.pkl')

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
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_POPE_qa_coco_vdd.pkl')

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
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_POPE_qa_aokvqa_vdd.pkl')

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
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_POPE_qa_gqa_vdd.pkl')

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
    res_df.to_pickle(f'{RESULT_PATH}/eval_res_mmvet_vdd.pkl')
    with open(f'{RESULT_PATH}/mmvet_vdd.json','w') as f:
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

    with open(f'{RESULT_PATH}/eval_res_mmmu_vdd.json', 'w') as f:
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
    with open(f'{RESULT_PATH}/eval_res_chair_vdd.json', 'w') as f:
        json.dump(results, f, indent=2)

from benchmarks.Llava_Bench import *
import shortuuid
def LLAVABENCH_eval():
    tokenizer, model, image_processor, context_len = load_llava_model()
    data_generator = load_dataset_llavabench()
    answers_file = os.path.expanduser(f"{RESULT_PATH}/eval_res_llavabench_vdd.jsonl")
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
    parser = argparse.ArgumentParser(description="VDD Llama evaluation")
    parser.add_argument("--benchmark_name", type=str,default="MME_Bench", help="benchmark name")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--use_dd", action='store_true', default=False)
    parser.add_argument("--use_dd_unk", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)

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





