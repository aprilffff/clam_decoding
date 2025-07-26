from models.config import *
from glob import glob
import os
from tqdm import tqdm
import json

def load_dataset_POPE_segmentation():
    image_paths=glob(DPATH_COCO+"/*")
    image_paths={p.aplit('/')[-1]:p for p in image_paths}

    annotation_path=DPATH_POPE+'/segamentation/coco_ground_truth_segmentation.json'
    with open(annotation_path,'r') as f:
        annotations=json.load(f)

    for ant in annotations:
        image_name=ant['image'].split('_')
        image_path=image_paths[image_name]
        target_obj=ant['objects']
        yield{'image_name':image_name,
              'image_path':image_path,
              'target_obj':target_obj}

def load_dataset_POPE_qa_coco():
    image_paths=glob(DPATH_COCO+"/*")
    image_paths={p.split('/')[-1]:p for p in image_paths}


    with open(DPATH_POPE+'/output/coco/coco_pope_random.json','r') as f:
        annotations_random = [json.loads(q) for q in f]

    with open(DPATH_POPE+'/output/coco/coco_pope_adversarial.json','r') as f:
        annotations_adversarial= [json.loads(q) for q in f]

    with open(DPATH_POPE + '/output/coco/coco_pope_popular.json', 'r') as f:
        annotations_popular = [json.loads(q) for q in f]

    annotations={"popular":annotations_popular,
                 "adversarial":annotations_adversarial,
                 "random":annotations_random}

    for a_name,a_dict in tqdm(annotations.items()):
        for ant in tqdm(a_dict,desc=f'POPE-coco-{a_name}'):
            image_path=image_paths[ant['image']]

            yield{'annotation_type':a_name,
                  'image_name':ant['image'],
                  'image_path':image_path,
                  'q':ant['text'],
                   'a':ant['label']}


def load_dataset_POPE_qa_aokvqa():
    image_paths = glob(DPATH_COCO + "/*")
    image_paths = {p.split('/')[-1]: p for p in image_paths}

    with open(DPATH_POPE + '/output/seem/aokvqa/aokvqa_pope_seem_random.json', 'r') as f:
        annotations_random = json.load(f)

    with open(DPATH_POPE + '/output/seem/aokvqa/aokvqa_pope_seem_adversarial.json', 'r') as f:
        annotations_adversarial =  json.load(f)

    with open(DPATH_POPE + '/output/seem/aokvqa/aokvqa_pope_seem_popular.json', 'r') as f:
        annotations_popular = json.load(f)

    annotations = {"popular": annotations_popular,
                   "adversarial": annotations_adversarial,
                   "random": annotations_random}

    for a_name, a_dict in annotations.items():
        for ant in tqdm(a_dict,desc=f'POPE-aokvqa-{a_name}'):
            image_path = image_paths[ant['image']]

            yield {'annotation_type': a_name,
                   'image_name': ant['image'],
                   'image_path': image_path,
                   'q':ant['text'],
                   'a':ant['label']}


def load_dataset_POPE_qa_gqa():
    image_paths = glob(DPATH_GQA + "/*")
    image_paths = {p.split('/')[-1]: p for p in image_paths}
    with open(DPATH_POPE + '/output/seem/gqa/gqa_pope_seem_random.json', 'r') as f:
        annotations_random = json.load(f)

    with open(DPATH_POPE + '/output/seem/gqa/gqa_pope_seem_adversarial.json', 'r') as f:
        annotations_adversarial = json.load(f)

    with open(DPATH_POPE + '/output/seem/gqa/gqa_pope_seem_popular.json', 'r') as f:
        annotations_popular = json.load(f)

    annotations = {"popular": annotations_popular,
                   "adversarial": annotations_adversarial,
                   "random": annotations_random}

    for a_name, a_dict in annotations.items():
        for ant in tqdm(a_dict,desc=f'POPE-gqa-{a_name}'):
            image_path = image_paths[ant['image']]
            yield {'annotation_type': a_name,
                   'image_name': ant['image'],
                   'image_path': image_path,
                   'q':ant['text'],
                   'a':ant['label']}