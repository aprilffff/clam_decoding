import os
import ast
import pandas as pd
from glob import glob
from tqdm import tqdm
from models.config import *
import numpy as np
import random
import re
import yaml

def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index) # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

def parse_img_path(text):
    matches = re.findall("<image (.*?)>", text)
    return matches

def process_single_sample(data):
    question = data.question
    o_imgs_paths = []
    for option in ast.literal_eval(data.options):
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)

    if len(o_imgs_paths) > 1:  # multiple images in options, used for random selection
        return {'id': data.id, 'question': question, 'options': data.options, 'answer': data.answer,
             'image': None, 'question_type': data.question_type}
    else:
        return {'id': data.id, 'question': question, 'options': data.options, 'answer': data.answer,
             'image': data.image_1['bytes'], 'question_type': data.question_type}

def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict
def construct_prompt(sample, config):
    question = sample['question']
    options = eval(sample['options'])
    example = ""
    if sample['question_type'] == 'multiple-choice':
        start_chr = 'A'
        prediction_range = []
        index2ans = {}
        for option in options:
            prediction_range.append(start_chr)
            example += f"({start_chr}) {option}\n"
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt_sample_structure = config['multi_choice_example_format'][0]
        empty_prompt = empty_prompt_sample_structure.format(question, example)
        res_dict = {}
        res_dict['index2ans'] = index2ans
        res_dict['correct_choice'] = sample['answer']
        res_dict['all_choices'] = prediction_range
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'][0].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt

        res_dict['gt_content'] = options[ord(sample['answer'].upper()) - ord('A')]
    else:
        empty_prompt_sample_structure = config['short_ans_example_format'][0]
        empty_prompt = empty_prompt_sample_structure.format(question)
        res_dict = {}
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'][0].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt
        res_dict['gt_content'] = sample['answer']

    res_dict.update(sample)
    return res_dict
def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    """

    start_chr = 'A'
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices
def load_dataset_MMMU_qa(root_dir=DPATH_MMMU+'/MMMU', split="validation"):
    """
    Args:
        root_dir: MMMU dataset root directory
        split: "test", "dev", or "validation"
    Yields:
        dict with fields: subject, image, q, choices, prompt, a
    """

    subjects = sorted(os.listdir(root_dir))
    config=load_yaml(DPATH_MMMU+'/llava1.5.yaml')
    for subject in subjects:
        subject_dir = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_dir):
            continue
        parquet_path = os.path.join(subject_dir, f"{split}-00000-of-00001.parquet")
        if not os.path.exists(parquet_path):
            continue

        df = pd.read_parquet(parquet_path)
        for row in tqdm(df.itertuples(index=False), desc=f"MMMU-{subject}-{split}",total=df.shape[0]):
            row=process_single_sample(row)
            row=construct_prompt(row,config)
            yield row



if __name__=='__main__':
    data= load_dataset_MMMU_qa()
    for d in data:
        a=1