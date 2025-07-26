from models.config import *
from glob import glob
import os
from tqdm import tqdm
import json

def load_dataset_mmvet():
    image_paths = glob(DPATH_MMVET + "/images/*")
    image_paths = {p.split('/')[-1]: p for p in image_paths}
    with open(DPATH_MMVET + '/mm-vet.json', 'r') as f:
        annotations = json.load(f)


    for ant_id,ant in tqdm(annotations.items(),desc="MMVet"):
        image_name = ant['imagename']
        image_path = image_paths[image_name]
        yield {'ant_id':ant_id,
                'image_name': image_name,
               'image_path': image_path,
               'q':ant['question'],
               'a':ant['answer'],
               'capability':ant['capability']}